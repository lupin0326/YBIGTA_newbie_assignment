#!/bin/bash

# miniconda가 존재하지 않을 경우 설치
if ! command -v conda &>/dev/null; then
    echo "Miniconda가 설치되어 있지 않음. 설치 중..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    echo "Miniconda 설치 완료!"
else
    echo "Miniconda가 이미 설치되어 있습니다."
fi

# Conda 환경 생성 및 활성화
if ! conda env list | grep -q "myenv"; then
    echo "가상환경 'myenv' 생성 중..."
    conda create -n myenv python=3.9 -y
    echo "가상환경 생성 완료!"
fi


## 건드리지 마세요! ##
python_env=$(python -c "import sys; print(sys.prefix)")
if [[ "$python_env" == *"/envs/myenv"* ]]; then
    echo "가상환경 활성화: 성공"
else
    echo "가상환경 활성화: 실패"
    exit 1 
fi

# 필요한 패키지 설치
pip install mypy

# Submission 폴더 파일 실행
cd submission || { echo "submission 디렉토리로 이동 실패"; exit 1; }

for file in *.py; do
    input_file="../input/${file%.py}_input"
    output_file="../output/${file%.py}_output"
    
    if [[ -f $input_file ]]; then
        echo "실행 중: $file"
        python "$file" < "$input_file" > "$output_file"
    else
        echo "입력 파일 없음: $input_file"
    fi
done

# mypy 테스트 실행행
mypy *.py

# 가상환경 비활성화
conda deactivate
