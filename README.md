# Mandelbrot-CPU

A coursework for *Assembly Language and Microcomputer Interface*. A small demo of Mandelbrot Set Visualization, running at about 15-40 FPS. 
More details in attached report.

## Usage

鼠标左键拖动；
键盘 * / 缩放，+ - 增加细节，0 1 2 3 4 选择不同渲染实现

## dependency

VS2017 IDE 提供的

- OpenMP 2.0
- Intel SIMD intrinsics
- NuGet 程序包

```xml
  <package id="nupengl.core" version="0.1.0.1" targetFramework="native" />
  <package id="nupengl.core.redist" version="0.1.0.1" targetFramework="native" />
```
