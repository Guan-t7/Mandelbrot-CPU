#include <stdlib.h>

#include <array>
#include <cctype>
#include <chrono>
#include <future>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include <gl/glut.h>
#include <omp.h>
#include <immintrin.h>

#include <tinycolormap.hpp>

// timing
using namespace std::chrono;

// window 
constexpr auto wHeight = 720, wWidth = 1280;
GLubyte frame[wHeight][wWidth][3]{ 0 };
//! origin is at top-left of the window
struct Cursor
{
	int x, y;
} cursor{-1, -1};

// rendering
//! origin relative to bottom-left of the window
struct ComplexPlane
{
	double init_cr, init_ci;
	double cr_step, ci_step; // 对应1个pix
} plane {-2.5, -1.0, 3.5 / wWidth, 2.0 / wHeight };

auto max_iterations = 1024;
auto mode = 2;
alignas(16) int pFractal[wHeight][wWidth]{ 0 };

// helper: color the scene
decltype(std::make_unique<GLubyte[][3]>(0)) colors;

void init_colorindex()
{
	colors = std::make_unique<GLubyte[][3]>(3 * max_iterations);
	for (size_t i = 0; i < max_iterations; i++)
	{
		auto color = tinycolormap::GetColor(double(i) / max_iterations, tinycolormap::ColormapType::Turbo);
		colors[i][0] = color.ri();
		colors[i][1] = color.gi();
		colors[i][2] = color.bi();
	}
}
void colored()
{
	for (int i = 0; i < wHeight; i++)
	{
		for (int j = 0; j < wWidth; j++)
		{
			auto index = pFractal[i][j] - 1;
			auto color = colors[index];
			std::memcpy(frame[i][j], color, 3 * sizeof(color[0]));
		}
	}
}

// render mode 0
int iterations(const double cr, const double ci)
{
	double zr = 0, zi = 0;
	int i = 0;
	while (zr * zr + zi * zi < 4.0 && i < max_iterations)
	{
		double re, im;
		re = zr * zr - zi * zi + cr;
		im = 2 * zr * zi + ci;
		zr = re;
		zi = im;
		i++;
	}
	return i;
}

void naive(const ComplexPlane p)
{
	auto ci = p.init_ci;
	for (int i = 0; i < wHeight; i++)
	{
		auto cr = p.init_cr;
		for (int j = 0; j < wWidth; j++)
		{
			pFractal[i][j] = iterations(cr, ci);
			cr += p.cr_step;
		}
		ci += p.ci_step;
	}
}

// mode 1
void with_omp(const ComplexPlane p)
{
#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < wHeight; i++)
	{
		auto ci = p.init_ci + i * p.ci_step;
		for (int j = 0; j < wWidth; j++)
		{
			auto cr = p.init_cr + j * p.cr_step;
			pFractal[i][j] = iterations(cr, ci);
		}
	}
}

// mode 2
void with_cppmt(const ComplexPlane p)
{
	// 4 pix row as a task
	constexpr auto NTASKS = wHeight / 4;
	std::array<std::future<void>, NTASKS> tasks{};
	for (size_t k = 0; k < NTASKS; k++)
	{
		auto this_i = k * wHeight / NTASKS, next_i = (k + 1) * wHeight / NTASKS;
		auto this_ci = p.init_ci + this_i * p.ci_step;
		auto fut = std::async([=]() {
			auto ci = this_ci;
			for (size_t i = this_i; i < next_i; i++)
			{
				auto cr = p.init_cr;
				for (size_t j = 0; j < wWidth; j++)
				{
					pFractal[i][j] = iterations(cr, ci);
					cr += p.cr_step;
				}
				ci += p.ci_step;
			}
		});
		tasks[k] = std::move(fut);
	}
	for (size_t k = 0; k < NTASKS; k++)
	{
		tasks[k].wait();
	}
}

// mode 3
void avx_inner_loop(int index, double cr, double ci, double cr_step)
{
	const auto ci_4pd = _mm256_set1_pd(ci);
	const auto four_4pd = _mm256_set1_pd(4.0);
	auto cr_4pd = _mm256_set_pd(cr + 3 * cr_step, cr + 2 * cr_step, cr + cr_step, cr);
	const auto foursteps_4pd = _mm256_set1_pd(4 * cr_step);
	for (int j = 0; j < wWidth; j+=4)
	{
		auto one_4pi = _mm256_set1_epi64x(1LL);
		// double zr = 0, zi = 0;
		auto zr_4pd = _mm256_setzero_pd(), 
			zi_4pd = _mm256_setzero_pd();
		// max_iterations control
		int i = 0;
		// bookkeeping i for 4 pix; i64
		auto i_4pi = _mm256_setzero_si256();
		// flow control
		auto cont_mask = _mm256_set1_epi64x(-1);
		while (i < max_iterations)
		{
			// double re, im;
			auto zr2_4pd = _mm256_mul_pd(zr_4pd, zr_4pd),
				zi2_4pd = _mm256_mul_pd(zi_4pd, zi_4pd);
			// while (zr * zr + zi * zi < 4.0)
			auto norm2_4pd = _mm256_add_pd(zr2_4pd, zi2_4pd);
			cont_mask = _mm256_castpd_si256(_mm256_cmp_pd(norm2_4pd, four_4pd, _CMP_LT_OQ));
			int all_done = _mm256_testz_si256(cont_mask, cont_mask);
			if (all_done) break;
			// re = zr * zr - zi * zi + cr;
			auto re_4pd = _mm256_sub_pd(zr2_4pd, zi2_4pd);
			re_4pd = _mm256_add_pd(re_4pd, cr_4pd);
			// im = 2 * zr * zi + ci;
			auto im_4pd = _mm256_mul_pd(zr_4pd, zi_4pd);
			im_4pd = _mm256_add_pd(im_4pd, im_4pd);
			im_4pd = _mm256_add_pd(im_4pd, ci_4pd);
			// zr = re;
			zr_4pd = re_4pd;
			// zi = im;
			zi_4pd = im_4pd;
			// i++;
			i++;
			// i++ for 4 pix respectively and conditionally
			one_4pi = _mm256_and_si256(one_4pi, cont_mask);
			i_4pi = _mm256_add_epi64(i_4pi, one_4pi);
		};
		// downcast from i64 to i32, shuffled to lower 64b in each 128b lane
		i_4pi = _mm256_shuffle_epi32(i_4pi, _MM_SHUFFLE(3, 1, 2, 0));
		// permute 4 * 64b s.t. lower 64b in 128b lane goes to lower 128b of YMM
		i_4pi = _mm256_permute4x64_epi64(i_4pi, _MM_SHUFFLE(3, 1, 2, 0));
		auto i_4pi32 = _mm256_castsi256_si128(i_4pi);
		// pFractal[index][j] = iterations(cr, ci);
		_mm_store_si128(reinterpret_cast<__m128i*>(&pFractal[index][j]), i_4pi32);
		// cr += cr_step;
		cr_4pd = _mm256_add_pd(cr_4pd, foursteps_4pd);
	}
}

void with_avx(const ComplexPlane p)
{
	auto ci = p.init_ci;
	for (int i = 0; i < wHeight; i++)
	{
		auto cr = p.init_cr;
		avx_inner_loop(i, cr, ci, p.cr_step);
		ci += p.ci_step;
	}
}

// mode 4
void avx_cppmt(const ComplexPlane p)
{
	// 4 pix row as a task
	constexpr auto NTASKS = wHeight / 4;
	std::array<std::future<void>, NTASKS> tasks{};
	for (size_t k = 0; k < NTASKS; k++)
	{
		auto this_i = k * wHeight / NTASKS, next_i = (k + 1) * wHeight / NTASKS;
		auto this_ci = p.init_ci + this_i * p.ci_step;
		auto fut = std::async([=]() {
			auto ci = this_ci;
			for (size_t i = this_i; i < next_i; i++)
			{
				auto cr = p.init_cr;
				avx_inner_loop(i, cr, ci, p.cr_step);
				ci += p.ci_step;
			}
		});
		tasks[k] = std::move(fut);
	}
	for (size_t k = 0; k < NTASKS; k++)
	{
		tasks[k].wait();
	}
}

// evaluation
void draw_string(std::string s)
{
	const char* c = s.c_str();
	while (*c)
	{
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *c++);
	}
}

void redraw()
{
	const std::string s_methods[] = { "0) naive", "1) omp", "2) cppmt", "3) avx", "4) avx_cppmt" };
	decltype(&naive) methods[] = { naive, with_omp, with_cppmt, with_avx, avx_cppmt };

	auto begin = steady_clock::now();
	methods[mode](plane);
	auto rendered = steady_clock::now();
	colored();
	auto colored = steady_clock::now();

	duration<double> iterTime = rendered - begin;
	duration<double> renderTime = colored - rendered;

	glViewport(0, 0, wWidth, wHeight);

	glMatrixMode(GL_PROJECTION);  // 选择投影矩阵
	glPushMatrix();               // 保存原矩阵
	glLoadIdentity();             // 装入单位矩阵
	glOrtho(0, wWidth, 0, wHeight, -1, 1);    // 位置正投影
	glMatrixMode(GL_MODELVIEW);   // 选择Modelview矩阵
	glPushMatrix();               // 保存原矩阵
	glLoadIdentity();             // 装入单位矩阵
	
	glRasterPos2i(0, 0);
	// display frame buffer
	glDrawPixels(wWidth, wHeight, GL_RGB, GL_UNSIGNED_BYTE, frame);
	// print msg on frame
	glColor3f(1.0, 1.0, 1.0);
	glRasterPos2i(0, 10);
	draw_string("Time Taken: " + std::to_string(iterTime.count()) + " + " +
		std::to_string(renderTime.count()) + "s");
	glRasterPos2i(0, 30);
	draw_string("Iterations: " + std::to_string(max_iterations));
	glRasterPos2i(0, 50);
	draw_string(s_methods[mode]);

	glMatrixMode(GL_MODELVIEW);   // 选择Modelview矩阵
	glPopMatrix();                // 重置为原保存矩阵
	glMatrixMode(GL_PROJECTION);  // 选择投影矩阵
	glPopMatrix();                // 重置为原保存矩阵

	glutSwapBuffers();
}

void mouse(int x, int y)
{
	cursor.x = x, cursor.y = y;
}

void pan(int x, int y)
{
	plane.init_cr -= plane.cr_step * (x - cursor.x);
	plane.init_ci += plane.ci_step * (y - cursor.y);
	cursor.x = x, cursor.y = y;
}

void key(unsigned char k, int x, int y)
{
	auto s = 0.9;
	switch (k)
	{
		case '+': 
			max_iterations += 64;
			init_colorindex();
			break;
		case '-': 
			if (max_iterations > 64)
			{
				max_iterations -= 64;
				init_colorindex();
			}
			break;
		case '/':
			s = 1 / s;
		case '*':
			plane.init_cr += (1 - s) * plane.cr_step * cursor.x; 
			plane.init_ci += (1 - s) * plane.ci_step * (wHeight - cursor.y);
			plane.cr_step *= s; plane.ci_step *= s;
			break;
		case 27:
		case 'q': {exit(0); break; }
		default:
			if (std::isdigit(k) && k < '5')
				mode = k - '0';
			break;
	}
}

void idle()
{
	glutPostRedisplay();
}

int main(int argc,  char *argv[])
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
	glutInitWindowSize(wWidth, wHeight);
	int windowHandle = glutCreateWindow("Mandelbrot");

	glutDisplayFunc(redraw);
	glutKeyboardFunc(key);
	glutPassiveMotionFunc(mouse);
	glutMotionFunc(pan);
	glutIdleFunc(idle);

	init_colorindex();

	glutMainLoop();
	return 0;
}