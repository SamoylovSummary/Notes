# Markdown

[TOC]

Обычный текст текст текст текст текст текст текст текст текст текст текст текст текст текст текст текст текст текст текст текст.

## Формулы

Язык разметки формул LaTeX. Online-редактор: http://www.codecogs.com/latex/eqneditor.php

### Стандартный способ, который не всегда работает

$$x=\frac{\sum_{i=1}^{n}w_{i}x_{i}}{\sum_{i=1}^{n}w_{i}}$$

$$\begin{pmatrix}
\alpha & \beta & \gamma\\
\sqrt[3]{2} & \lim_{x\rightarrow 0}x & \prod_{i=1}^n\frac{1}{i}
\end{pmatrix}$$

$$\vec{a}=
\begin{cases}
\vec{b}&\text{if } x\in \mathbb{N} \\
\vec{c}&\text{if } x\neq 0
\end{cases}$$

А это: $x=\frac{\sum_{i=1}^{n}w_{i}x_{i}}{\sum_{i=1}^{n}w_{i}}$ inline-вариант

### Обход проблемы для Github

<img src="https://latex.codecogs.com/gif.latex?$$x=\frac{\sum_{i=1}^{n}w_{i}x_{i}}{\sum_{i=1}^{n}w_{i}}$$" title="$$x=\frac{\sum_{i=1}^{n}w_{i}x_{i}}{\sum_{i=1}^{n}w_{i}}$$" />

<img src="https://latex.codecogs.com/gif.latex?$$\begin{pmatrix}&space;\alpha&space;&&space;\beta&space;&&space;\gamma\\&space;\sqrt[3]{2}&space;&&space;\lim_{x\rightarrow&space;0}x&space;&&space;\prod_{i=1}^n\frac{1}{i}&space;\end{pmatrix}$$" title="$$\begin{pmatrix} \alpha & \beta & \gamma\\ \sqrt[3]{2} & \lim_{x\rightarrow 0}x & \prod_{i=1}^n\frac{1}{i} \end{pmatrix}$$" />

<img src="https://latex.codecogs.com/gif.latex?$$\begin{pmatrix}&space;\alpha&space;&&space;\beta&space;&&space;\gamma\\&space;\sqrt[3]{2}&space;&&space;\lim_{x\rightarrow&space;0}x&space;&&space;\prod_{i=1}^n\frac{1}{i}&space;\end{pmatrix}$$" title="$$\begin{pmatrix} \alpha & \beta & \gamma\\ \sqrt[3]{2} & \lim_{x\rightarrow 0}x & \prod_{i=1}^n\frac{1}{i} \end{pmatrix}$$" />

<img src="https://latex.codecogs.com/gif.latex?$$\vec{a}=&space;\begin{cases}&space;\vec{b}&\text{if&space;}&space;x\in&space;\mathbb{N}&space;\\&space;\vec{c}&\text{if&space;}&space;x\neq&space;0&space;\end{cases}$$" title="$$\vec{a}= \begin{cases} \vec{b}&\text{if } x\in \mathbb{N} \\ \vec{c}&\text{if } x\neq 0 \end{cases}$$" />

А это: <img src="https://latex.codecogs.com/gif.latex?\inline&space;x=\frac{\sum_{i=1}^{n}w_{i}x_{i}}{\sum_{i=1}^{n}w_{i}}" title="x=\frac{\sum_{i=1}^{n}w_{i}x_{i}}{\sum_{i=1}^{n}w_{i}}" /> inline-вариант

## Форматирование

Список:
- *italic*
- **bold**
- ~~srikethrough~~
- ==highlight==
- ^superscript^
- ~subscript~
- `code`

Пропадающие спецсимволы: x*y*z

Код:

	x*y*z
	line 2

То же самое другим способом:

```
x*y*z
line 2
```

Таблица:

| Заголовок 1 | Заголовок 2 |
|-------------|-------------|
| ячейка 1.1  | ячейка 1.2  |
| ячейка 2.1  | ячейка 2.2  |

