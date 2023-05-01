#TS-Model

- 단순 벡터자기회귀 모형의 경우 앞서 변수의 나열순서에 따라 인과관계가 임의로 정해지며 이에 따라 외생성이 강한 변수를 먼저, 내생성이 강한 변수를 나중에 나열한다.
- 이를 위해 [Granger causality test]를 통해 변수간 외생성 여부(인과관계의 방향)를 파악하여 변수를 나열할 수 있지만, 인과관계가 양방향으로 작용할 경우 한계가 있다.
- 촐레스키 분해를 통해 얻은 오차항에는 회귀모형이 설명하지 못하는 모든 정보가 포함되어 있어 경제적 의미 부여하는 것이 불가능하다.


- SVAR: Structural Vector AutoRegressive 모형
> : 2 계열(변수) SVAR(1) 모형 가정  
> $$ 
 \begin{align}
	 &a_{11}y_{1,t} = a_{1}^{*}-a_{12}y_{2,t}+a_{11}^{*}y_{1,t-1}+a_{12}^{*}y_{2,t-1}+b_{11}\epsilon_{1,t}+b_{12}\epsilon_{2,t}   \\
	 &a_{22}y_{2,t} = a_{2}^{*}-a_{21}y_{1,t}+a_{21}^{*}y_{2,t-1}+a_{22}^{*}y_{2,t-1}+b_{21}\epsilon_{1,t}+b_{22}\epsilon_{2,t} \\ \\
	 &\rightarrow 	
		\begin{bmatrix}
			a_{11}\quad a_{12} \\
			a_{21}\quad a_{22}
		\end{bmatrix}
		\begin{bmatrix}
			y_{1,t} \\
			y_{2,t}
		\end{bmatrix}
		=
		\begin{bmatrix}
			a_{1}^{*} \\
			a_{2}^{*}
		\end{bmatrix}
		+
		\begin{bmatrix}
			a_{11}^{*}\quad a_{12}^{*} \\
			a_{21}^{*}\quad a_{22}^{*}
		\end{bmatrix}
		\begin{bmatrix}
			y_{1,t} \\
			y_{2,t}
		\end{bmatrix}
		+
		\begin{bmatrix}
			b_{11}\quad b_{12} \\
			b_{21}\quad b_{22}
		\end{bmatrix}
		\begin{bmatrix}
			\epsilon_{1,t} \\
			\epsilon_{2,t}
		\end{bmatrix} 
	\\ \\
	&\rightarrow
		Ay_{t}=a^{*}+A_{1}^{*}y_{t-1}+B\epsilon_{t}, \epsilon_{t} \sim  iid(0,I_{2})
		
 \end{align}
 $$
- VAR과 SVAR 의 관계
> SVAR모형에 A의 역행렬을 취해주면 VAR 모형으로 표현된다.
>  $$
  \begin{align}
	 &Ay_{t}=a^{*}+A_{1}^{*}y_{t-1}+\dots+A_{p}^{*}y_{t-p}+B\epsilon_{t} \quad;\quad SVAR(p) \\
	 &\Longrightarrow
	 y_{t}=A^{-1}a^{*}+A^{-1}A_{1}^{*}y_{t-1}+\dots+A^{-1}A_{p}^{*}y_{t-p}+A^{-1}B\epsilon_{t} \\
	 & \quad\quad\quad = a+A_{1}y_{t-1}+\dots+A_{p}y_{t-p}+u_{t} \quad;\quad VAR(p) \\
	 & where \quad A_{i} = A^{-1}A_{i}^{*},A^{-1}B\epsilon_{t} = u_{t}, \epsilon_{t} \sim iid(0,I), u_{t}\sim iid(0,\Omega) \\
	 &Notice that  \quad \Omega = (A^{-1}B)(A^{-1}B)' \quad and \quad A\Omega A' = BB' 
  \end{align}
 $$
- SVAR 모형의 형태 : 표기상 편의의 차이 분석의 결과는 같다.
>1) AB형 모형
>2) A형 모형
>3) B형 모형
- 제약모형
> K 변량 SVAR모형의 경우 충격반응분석을 위해 일반적으로 추정해야 할 모수 $K^{2}$ 개인데 분산행렬이 대칭이라 $K(K+1)/2$ 개의 식만 주어진다. 따라서 추정을 위해선 별도의 제약이 필요하다. 제약을 가할 땐 보통 경제적 이론에 근간을 둔다.
> 1) 단기제약(contemporaeous restrictions)모형
> 	  2변량 A형 모형을 통해 단기제약의 경우를 보면 (물가와 총수요 간의 관계)
> 	  $$
 	\begin{align}
		 	&A
		 	\begin{bmatrix}
			 	P_{t} \\
			 	gdP_{t}
			    \end{bmatrix}
			= a^{*} + A_{1}^{*}
		 	 \begin{bmatrix}
			 	P_{t-1} \\
			 	gdP_{t-1}
			    \end{bmatrix}
			+
		 	 \begin{bmatrix}
			 	\mathcal{E}_{P,t} \\
			        \mathcal{E}_{gdP,t}
			    \end{bmatrix}	\\ \\
			&Let \quad A^{-1} =
			    \begin{pmatrix}
			 	\mathcal{\alpha}_{11} \quad \mathcal{\alpha}_{12} \\
			        \mathcal{\alpha}_{21} \quad \mathcal{\alpha}_{22}
			    \end{pmatrix} ,	\\
			    &then \quad \Omega=
			    \begin{pmatrix}
			 	\omega_{11} \quad \omega_{12} \\
			        \omega_{21} \quad \omega_{22}
			    \end{pmatrix} = 
			    \begin{pmatrix}
			 	\mathcal{\alpha}_{11} \quad \mathcal{\alpha}_{12} \\
			        \mathcal{\alpha}_{21} \quad \mathcal{\alpha}_{22}
			    \end{pmatrix}
			    \begin{pmatrix}
			 	\mathcal{\alpha}_{11} \quad \mathcal{\alpha}_{12} \\
			        \mathcal{\alpha}_{21} \quad \mathcal{\alpha}_{22}
			    \end{pmatrix}	'
			    =
			    A^{-1}A^{-1'} \\ \\	
			& \omega_{11} = \alpha_{11}^{2} + \alpha_{12}^{2} \\
			& \omega_{12} = \alpha_{11}\alpha_{21} + \alpha_{12}\alpha_{22} = \omega_{21} \\
			& \omega_{22} = \alpha_{21}^{2} + \alpha_{22}^{2} 		
		 \end{align}  
 	 
      $$
>   위와 같이 모형은 추정해야할 모수 $\alpha$가 4개이지만 방정식은 3만 주어져 있다.
>   이때 케인즈학파의 이론을 바탕으로 단기에 총수요 충격이 즉각적으로 물가에 영향을 줄 수 없다고 가정하면 단기 제약 $\alpha_{12}=0$ 을 부여해 모수를 추정할 수 있게 된다.
>   
> 2) 