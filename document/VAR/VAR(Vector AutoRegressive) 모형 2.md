#TS-Model  #VAR

> [[Reference_list]]
> STATA를 이용한 응용계량경제학. 박승록(2020.01.20)

- 만약 여러 변수들 사이에 동시적 인과관계(연립성)이 있다면, 이 변수들은 모두 동일하게 취급되므로 내생변수와 외생변수의 구분이 불필요
-  각 방정식은 동일한 독립변수에 의해 설명되는 유도형 함수(Reduced form equation)로 표기 가능 $\Longrightarrow$ 벡터자기회귀모형(Vector Autoregressive: VAR)으로 발전

- SVAR을 식으로 표현
$$ 
\begin{align} 
	&Y_{t}=\beta_{10}-\beta_{12}X_{t}+\gamma_{11}Y_{t-1}+\gamma_{12}X_{t-1}+\upsilon_{yt} \\ \\
	&X_{t}=\beta_{20}-\beta_{21}Y_{t}+\gamma_{21}Y_{t-1}+\gamma_{22}X_{t-1}+\upsilon_{t2} 
\end{align} 
\quad \Longrightarrow (1)
$$
여기서 $Y_{t}$와 $X_{t}$ 모두 안정적 시계열자료, $\upsilon_{yt},\upsilon_{xt}$는 상관되지 않은 백색잡음과정으로 행렬로 표기하면,
$$
\begin{align}
	&\begin{bmatrix}
		1 \quad \beta_{12} \\
		\beta_{21} \quad 1
	\end{bmatrix}
	\begin{bmatrix}
		Y_{t} \\
		X_{t}		
	\end{bmatrix}
	=
	\begin{bmatrix}
		\beta_{10} \\
		\beta_{20}
	\end{bmatrix}
	+
	\begin{bmatrix}
		\gamma_{11} \quad \gamma_{12} \\
		\gamma_{21} \quad \gamma_{22}
	\end{bmatrix}
	\begin{bmatrix}
		Y_{t-1} \\
		X_{t-1}		
	\end{bmatrix}
	+
	\begin{bmatrix}
		\upsilon_{yt} \\
		\upsilon_{xt}
	\end{bmatrix}
	\\ \\
	&B Z_{t}= \Gamma_{0} + \Gamma_{1}Z_{t} + \upsilon_{t}
\end{align}
	
$$
여기서  $B^{-1}$을  행렬 연산하여 B를 제거하고 $Z_{t}$에 관하여 풀면
$$
	Z_{t}=B^{-1}\Gamma_{0}+B^{-1}\Gamma_{1}Z_{t-1}+B^{-1}\upsilon_{t} = A_{0} + A_{1}Z_{t-1} + e_{t}
$$
이를 대수적 표현으로 나타내면 다음과 같은 표준적 VAR 모형이 됨
$$
\begin{align}
	&Y_{t}=\alpha_{10} + \alpha_{11}Y_{t-1}+\alpha_{12}X_{t-1}+e_{yt} \\
	&X_{t}=\alpha_{20} + \alpha_{21}Y_{t-1}+\alpha_{22}X_{t-1}+e_{xt}
\end{align}
$$
여기서 오차항, $e_{yt},e_{xt}$는 $e_{t}=B^{-1}\upsilon_{t}$이므로 구조적 VAR(SVAR)모형의 서로 상관관계가 없는 오차항 $\upsilon_{yt},\upsilon_{xt}$와 현시점에서 종속변수간의 관계를 나타내는 파라미터가 결합된 것
$$
	\begin{align}
		&e_{xt}=(\upsilon_{yt}+\beta_{12}\upsilon_{xt}/(1-\beta_{12}\beta_{21})) \\
		&e_{yt}=(\upsilon_{xt}+\beta_{21}\upsilon_{xt}/(1-\beta_{12}\beta_{21}))
	\end{align}
$$

- 일반 VAR
> 장점
1) 외생변수, 내생변수의 구분 불필요
2) 유도형 모형(reduced form model)이므로 개별 방정식에 고전적 최소자승법 적용 가능
3) 좋은 예측치 제공
> 단점
1) 경제이론 경시 $\longrightarrow$ 대안으로 추정후 인과관계의 검정을 통해 이론적 기반 반영 가능
2) 추정과정에서 많은 자유도 상실
3) 파라미터 추정치의 해석에 어려움 $\longrightarrow$ 대안으로 충격반응함수
-  VAR 모형은 바로 구조모형의 유도형 모형이고 오차항 역시 유도된 형태인 구조모형의 오차항들의 결합체이므로 어떤 방법으로든 구조모형의 오차를 구분하는 것이 필요한데 이를 VAR모형에선의 식별문제(identification problem)라고 한다.