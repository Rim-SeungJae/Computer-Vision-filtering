<img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=Python&logoColor=white"/> <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=OpenCV&logoColor=white"/>
# 소개
이 저장소는 OpenCV를 활용한 몇가지 간단한 프로젝트를 담고 있습니다.

## A1_image_filtering.py
이미지 혹은 커널 사이의 cross-correlation을 구하는 함수 cross_correlation_1d, cross_correlation_2d와 필터의 크기와 시그마 값을 입력받아 해당하는 가우시안 필터를 반환하는 함수 get_gaussian_filter_1d,get_gaussian_filter_2d를 구현했다.
이를 활용하여 lenna.png와 shapes.png에 여러가지 가우시안 필터를 적용해 보고 또 2차원 필터를 한번에 적용했을 때와 해당 2차원 필터를 1차원 필터 두개로 분해한 필터를 각각 적용했을 때의 차이를 비교해 보았다.
그 결과 2차원 필터를 한번에 적용하는 것보다 분해된 1차원 필터를 각각 적용했을 때의 속도가 월등히 빠르다는 사실을 알 수 있었다(그러나 필터를 적용한 결과가 정확하게 일치하지는 않았다).

## A2_edge_detection.py
소벨필터를 활용해 이미지의 변화값을 구하는 함수 compute_image_gradient 및 비최대값 억제를 수행하는 함수 non_maximum_suppression_dir를 구현하였다.
이를 앞서 구현했던 함수들과 함께 활용하여 lenna.png와 shapes.png에 존제하는 edge를 감지하는 작업을 수행하였다.

## A2_corner_detection.py