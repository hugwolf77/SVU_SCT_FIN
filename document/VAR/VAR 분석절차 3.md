#TS-Model #VAR


#### VAR 분석 절차
1) 시계열의 안정성 검정
    - 상관계수 분석
    - 그래프를 통한 직관적 확인
    -  DF, ADF test 등 unit root test 실행 (평균이 0이 아닌 증가율 데이터 같은 경우 상수항 옵션이 필요, 시차가 있는 경우에는 ADF)
2) VAR모형의 예비분석
3) VAR모형의 차수결정
4) 모형의 추정
5) 진단
	a. 잔차의 자기상관 여부 검정
	b. 모형의 안정성
	c. 잔차의 정규성
	d. 적성시차의 선택
    e. 인과성 검정 Granger
6) 인과성
 =>  Granger Causality test로 변수의 순서(Ordering)결정 + Cross Covariance Function
 =>
7) 충격반응함수
8) 예측오차분산의 분해
9) 예측