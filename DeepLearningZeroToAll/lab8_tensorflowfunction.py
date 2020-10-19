# shape, rank(맨 처음 괄호 수), axis(세로=0, 가로=1, -1)
# matmul vs multiply - broadcasting 때문에 그냥 *을 하여 곱하면 값이 이상해짐
# Reduce mean: 평균 계산 (정수 적으면 int 결과, 1., 2. 등 .을 끝에 붙여야 float 결과)
# Argmax: 가장 큰 원소의 위치
# Reshape: 모양 바꾸기, squeeze: 차원 줄이기 expand_dims: 차원 추가
# One hot: 숫자값을 자리로 0, 1 표현
# Casting: 원하는 data type으로 변형
# Stack: 배열을 차례대로 쌓음
# ones_like, zeros_like: 같은 모양으로 1/0 채움
# for x,y in zip([1,2,3],[4,5,6]):, zip: 배열을 하나씩 가져오는 것
