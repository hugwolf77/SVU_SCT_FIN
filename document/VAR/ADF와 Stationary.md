<span style="color:#BFFD9F"> 2023.04.19</span>
[[Finance]]
#Finance 

- ### ADF(Augmented Dicky Fuller)검정
	: 정상성(_stationary_)을 알아보기 위한 단위근 검정(unit root test: ARIMA 모형의 적분 차수를 판단), Dicky Fuller 검정에 시차  차분(lagged differences)를 더해준다.	적정한 시간 간격 (time lag)가 존재한다.
	
	1) 상수항도 없고, 추세도 없는 경우  $$ \Delta y_t = Y_{y_{t-1}} + \sum_{s=1}^{m}a_{s}\Delta y_{t-s} + v_{t}  $$
	2) 상수항은 있고, 추세는 없는 경우 $$\Delta y_t = \alpha + Y_{y_{t-1}} + \sum_{s=1}^{m}a_{s}\Delta y_{t-s} + v_{t}$$
	3) 상수항도 있고, 추세도 있는 경우 $$ \Delta y_t = Y_{y_{t-1}} + \lambda_{t} + \sum_{s=1}^{m}a_{s}\Delta y_{t-s} + v_{t}  $$
- ### Stationary
	: 정상성에도 Strongly와 Weakly 로 나눌 수 있다.
	(1) Strong Stationarity
		1) $Y_{t},t \ge 1$에 대해서 $(Y_{1},\cdots,Y_{m})$ & $(Y_{1+k},\cdots,Y_{m+k})$  두 시계열이 동일한 결합확률분포를 가진다.
	 2) 기대치/평균과 분산이 일정하다. (시간과 독립적이다.)
	 3) 자기공분산/자기상관계수가 time lag 시간 간격에만 의존한다. 
	(2) Weak Stationarity
	 1) 기대치/평균과 분산이 일정하다.
	 2) 자기공분산/자기상관계수가  time lag시간 간격에만 의존한다.
	 3) $Y_{t},t \ge 1$에 대해서 $(Y_{1},\cdots,Y_{m})$ & $(Y_{1+k},\cdots,Y_{m+k})$  두 시계열의 결합확률분포가 다변량정규분포를 따르면, 강정상성과 약정상성은 일치한다.
- ### Test for stationarity
	1) 시계열 그래프를 통해 패턴(평균회귀, 변동폭 등) 파악
	2) ACF 자기상관함수 그래프 감소 패턴 파악
	3) 단위근 검정 : 시계열의 단위근 포함 여부를 가설 검정

- 환율변동이 한국 5대 신산업의 중국수출입상품에 미치는 영향 분석-원달러, 엔달러, 위안화달러환율을 중심으로 
- 손 상 모
- 목포대학교 대학원
- 금융보험학협동과정 금융전공

- 생산성 충격이 실질환율과 순수출에  미치는 영향의 국별 비교 
- 박 미 숙 
- 경제학부 경제학 전공 
- 서울대학교 대학원