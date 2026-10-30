# gpu_bench.sh

합성 이미지로 **CNN 학습(순전파 → 역전파 → Adam)**을 반복하는 Bash 스크립트입니다. 외부 데이터셋 없이 GPU 사용률·메모리·온도·전력을 기록합니다.

NVIDIA 드라이버 외에 CUDA 지원 PyTorch가 필요합니다. 아래는 CUDA 12.6 빌드 설치 예시이며, 환경에 맞는 빌드는 [PyTorch 공식 설치 페이지](https://pytorch.org/get-started/locally/)에서 선택하세요.

```bash
sudo apt-get install -y python3-venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
```

파일을 Ubuntu에 옮긴 후 실행합니다.

```bash
# GPU 0번에서 60초 동안 테스트
GPU=0 DURATION=60 BATCH=32 IMAGE_SIZE=224 bash gpu_bench.sh

# 부하와 메모리 사용량 증가
GPU=0 DURATION=120 BATCH=128 IMAGE_SIZE=384 bash gpu_bench.sh

# 메모리 부족 시 배치 축소
BATCH=8 bash gpu_bench.sh
```

실행 결과는 다음과 같습니다.

- `gpu.csv`: 1초 간격 GPU 상태 기록
- `train.log`: 처리량(images/s), 학습 시간, PyTorch 최대 allocated/reserved 메모리
- 화면 출력: 평균·최대 GPU 사용률, 최대 VRAM 사용량

`nvidia-smi` 메모리는 다른 프로세스를 포함한 **GPU 전체 사용량**입니다. ([NVIDIA 지표 설명](https://docs.nvidia.com/deploy/nvidia-smi/)) 처리량은 워밍업을 제외하며, GPU 사용률 평균은 초기화와 워밍업을 포함합니다.

Bash/Python 문법과 프로세스 종료 처리를 검증했습니다. 실제 GPU 실행은 미검증입니다.