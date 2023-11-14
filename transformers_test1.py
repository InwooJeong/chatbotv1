from transformers import AutoTokenizer
from transformers import TFGPT2LMHeadModel
import tensorflow as tf

repo = 'skt/kogpt2-base-v2'  # 허깅페이스에 등록된 사전 학습된 모델 레파지토리
tokenizer = AutoTokenizer.from_pretrained(repo)
model = TFGPT2LMHeadModel.from_pretrained(repo, from_pt=True)

# 다음에 나올 토큰 랜덤하게 처리하여 문장을 생성
import numpy as np
import random

text = '공부를 잘하기 위해서는'
input_vector = tokenizer.encode(text)

while len(input_vector) < 30:   # 제시한 문장 + 새로 만들어지는 문장 => 30 토큰이면 종료
  # 1. 모델 예측 => 원문 + 예측 토큰(계속 누적) => 보다 완벽한 문장을 유도 단, 랜덤이 성능에 영향을 미침
  #                 최초 입력 문장을 계속 입력 => 연산 비용은 상승, 장기기억 분야에서는 도움
  output = model(np.array([input_vector]))
  # 2. 다음 토큰으로 나올 예측값들 중 탑 5 추출
  top5 = tf.math.top_k(output.logits[0,-1],k=5)
  # 3. 이 중에서 랜덤으로 1개 추출
  token_id = random.choice(top5.indices.numpy())
  # 4. 해당 글자를 출력 -> 클라이언트에게 전송(바로 혹은 3-4개씩 묶어서)
  print(tokenizer.decode(token_id),end=' ')
  # 5. 입력 벡터에 추가
  input_vector.append(token_id)