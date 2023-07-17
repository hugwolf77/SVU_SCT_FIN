## #TS-Model #VAR

papersize: a4
mainfont: 바탕체
margin-left: 0.5in
margin-right: 0.5in
margin-bottom: 0.5in
margin-top: 0.5in

---

### VAR model의 수식 이해

- 크리스토퍼 심스 교수 1980년 VAR 모형 개발
  1.  classical한 가정을 충족할 경우에만 단순회귀나 다중회귀 모형 또는 연립방정식 모델로 분석가능한 것을 내/외생 변수의 구분 없이 적용할 수 있는 다변량 시계열모형인 VAR 로 적용할 수 있게 되었음.
  2.  VAR은 다른 변수의 변동에 시차까지 고려하여 변동을 설명할 수 있는 모형으로
  3.  Granger Causality 분석과 함께 인과구조 설명하고 그 크기도 분석 가능하다.

Y, M 두 계열이 존재한다고 가정하고 시차는 2차까지만 고려 했을 경우
다음과 같은 식을 고려할 수 있게 된다.

\begin{align}
&Y*{t}=c*{1}+\alpha*{11}Y*{t-1}+\alpha*{12}Y*{t-2}+\beta*{11}M*{t-1}+\beta*{12}M*{t-2}+\epsilon*{t1} \\
&M*{t}=c*{2}+\alpha*{21}Y*{t-1}+\alpha*{22}Y*{t-2}+\beta*{21}M*{t-1}+\beta*{22}M*{t-2}+\epsilon*{t2}
\end{align}

이를 Vector 형식으로 다루기 위해 행렬 대수 형태로 바꾸면

\begin{equation}
X*{t}=
\begin{pmatrix}
Y*{t} \\
M*{t}
\end{pmatrix},
c =
\begin{pmatrix}
c*{1} \\
c*{2}
\end{pmatrix},
A*{1} =
\begin{pmatrix}
\alpha*{11}\quad\beta*{11} \\
\alpha*{21}\quad\beta*{21}
\end{pmatrix},
A*{2} =
\begin{pmatrix}
\alpha*{12}\quad\beta*{12} \\
\alpha*{22}\quad\beta*{22}
\end{pmatrix},
\epsilon*{t}
\begin{pmatrix}
\epsilon*{t1} \\
\epsilon*{t2}
\end{pmatrix}
\end{equation}

이를 통해 (1)식을 간단히 표현하면 다음과 같다.
$$ X*{t}=c+A*{1}X*{t-1}+A*{2}X*{t_2}+\epsilon*{t}$$

### VAR 모형 충격반응 함수

- 위에서 우리는 2개의 계열을 다뤘다. 따라서 위에 모형에서는 2개의 충격 모형이 가능하다.
- 따라서 이 두개 계열의 $\epsilon_{t1}$과 $\epsilon_{t2}$를 통해서 충격을 설정할 수 있다.
- 이때 각각 자신에 대한 충격과 다른 계열에 대한 영향으로 총 4개의 충격을 고려할 수 있다.

1. 먼저, 두 계열의 충격 간에 영향 관계가 없다는 즉, 충격이 2개만 존재한다는 가정으로
   먼저 Y 계열에 의한 충격 을 정의하여 표현하면
   $$\epsilon_{t1} =  \sigma_{y}$$
   으로 t1 시점에서 $\sigma_{y}$ 표준편차 만큼의 충격이 가해졌다는 의미고 innovation이라고 말한다.
   기본적으로 base 원점과 다른 M 계열의 충격이 없으며, 오차도 없다고 가정한다.

   $$
   \epsilon_{t2}=0 \quad Y_{0}=0,\quad M_{0}=0
   $$

   이는 결국 초기 절편이 없다는 뜻이며, 충격에 대해서 t 시점의 변화에 따라 수식적 변화를 보면,

   \begin{align}
   t = 1 &\rightarrow Y*{1}=\epsilon*{t1}=\sigma*{y} \\
   &\rightarrow M*{1}=\epsilon*{t2}=0 \\ \\
   t = 2 &\rightarrow Y*{2}=\alpha*{11}Y*{1}+\alpha*{12}Y*{0}+\beta*{11}M*{1}+\beta*{12}M*{0}= \alpha*{11}\sigma*{y} \\
   &\rightarrow M*{2}=\alpha*{21}Y*{1}+\alpha*{22}Y*{0}+\beta*{21}M*{1}+\beta*{22}M*{0}= \alpha*{21}\sigma\_{y}
   \end{align}

   여기서 다시 t =3 를 살펴 보면

   \begin{align}
   t = 3 &\rightarrow Y*{3}=\alpha*{11}Y*{2}+\alpha*{12}Y*{1}+\beta*{11}M*{2}+\beta*{12}M*{1}= \sigma*{y}(\alpha*{11}^2+\alpha*{12}+\beta*{11}\alpha*{21}) \\
   &\rightarrow M*{3}=\alpha*{21}Y*{2}+\alpha*{22}Y*{1}+\beta*{21}M*{2}+\beta*{22}M*{1}= \sigma*{y}(\alpha*{21}\alpha*{11}+\alpha*{22}+\beta*{21}\alpha\_{21})
   \end{align}

   이때 표준편차와 계수 모두가 1보다 작으므로 시간이 흐를 수록 미치는 영향은 0으로 수렴하게 됨.

2. 충격반응함수는 변수간의 인과관계를 파악하고 충격의 영향을 분석
3. 충격반응함수는 vAR 모형의 추정계수를 이용, 모형내 특정 변수에 대해 일정한 크기의 충격을 가할 때 모형내 변수들이 시간이 흐름에 따라 어떻게 반응하는지를 분석
4. 그러나 위에 가정과 다르게 현실에서는 Y와 M의 충격이 독립적이지 않기 때문에 하나의 변수가 충격을 받으면 다른 변수에 영향을 주고 이것이 다시 환류(feedback)되어 처음 충격이 시작된 변수에 영향을 주게 됨. 이러한 효과는 원래 충격에 대한 크기에 영향을 주기 때문에 이를 분해를 통해 제거해 주어야 한다. 보통 충격요인에 대한 공분산 행렬 직교화라고 한다. VAR 에서 촐레스키 분해 (Choleski factorization)를 사용한다.

   \begin{align}
   &Y*{t}=c*{1}+\alpha*{11}Y*{t-1}+\alpha*{12}Y*{t-2}+\beta*{11}M*{t-1}+\beta*{12}M*{t-2}+\epsilon*{t1}
   \\
   &M*{t}=c*{2}+\alpha*{21}Y*{t-1}++\alpha*{22}Y*{t-2}+\beta*{21}M*{t-1}+\beta*{22}M*{t-2}+\gamma\epsilon*{t1}+\epsilon\_{t2}
   \end{align}

5. 예측오차분산분해 (Forecast Error Variance Decomposition)
   특정변수 움직임에 대한 설명을 사용하여 각 변수의 상대적인 중요성을 파악하는 방법, 예측오차에서 여러 변수들의 충격을 포함하기 때문에 이를 전체 충격 크기에 대한 분해를 통해서 영향관계의 중요도를 분석할 수 있다.
