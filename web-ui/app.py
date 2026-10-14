from flask import Flask, render_template, jsonify, Response
import random
import time
import json
import uuid

app = Flask(__name__)

class EMACoreProcessor:
    def __init__(self):
        self.is_vision_active = False
        self.power_baseline = 0.185
        self.power_active = 2.642
        self.metrics = {
            "node_id": "ARM-CORTEX-A55-V12",
            "kernel": "EMA-FSL-RUNTIME-5.4.0",
            "quantization": "INT8-Asymmetric"
        }

    def generate_telemetry_stream(self):
        while True:
            v_vec = round(random.uniform(0.12, 0.28), 3)
            d_vec = round(random.uniform(2.15, 2.28), 2)
            temp = round(random.uniform(41.8, 43.2), 1)
            
            payload = {
                "radar": {
                    "v_vector": v_vec,
                    "d_vector": d_vec,
                    "rssi": random.randint(-68, -62),
                    "phase_noise": round(random.uniform(0.001, 0.004), 4)
                },
                "hardware": {
                    "cpu_util": random.randint(14, 19) if not self.is_vision_active else random.randint(78, 85),
                    "temp": temp,
                    "mem_used": random.randint(214, 218),
                    "power_w": self.power_baseline if not self.is_vision_active else self.power_active
                },
                "latency": "0.12ms",
                "timestamp": time.strftime("%H:%M:%S.%f")[:-3]
            }
            yield f"data:{json.dumps(payload)}\n\n"
            time.sleep(0.5)

core = EMACoreProcessor()

@app.route('/')
def main_dashboard():
    return render_template('index.html')

@app.route('/stream_telemetry')
def stream_telemetry():
    return Response(core.generate_telemetry_stream(), mimetype='text/event-stream')

@app.route('/invoke_vision_inference', methods=['POST'])
def invoke_vision_inference():
    core.is_vision_active = True
    time.sleep(0.68) # 模拟 CNN-TCN 空间时序特征提取耗时
    
    events = [
        {
            "event_id": str(uuid.uuid4())[:8],
            "type": "跌倒危险判定",
            "class": "CRITICAL",
            "confidence": 0.924,
            "inference_time": "6.8ms",
            "description": "检测到重心轨迹垂直分量突变，Triplet-Loss 特征距离突破警戒阈值"
        },
        {
            "event_id": str(uuid.uuid4())[:8],
            "type": "行为复核-正常",
            "class": "INFO",
            "confidence": 0.895,
            "inference_time": "6.5ms",
            "description": "时序特征符合‘快速坐下’模式，多模态协同判定为非风险动作"
        },
        {
            "event_id": str(uuid.uuid4())[:8],
            "type": "呼吸微动监测",
            "class": "WARNING",
            "confidence": 0.812,
            "inference_time": "7.1ms",
            "description": "视觉语义识别无大范围位移，雷达点云检测到胸腔微动频率异常减弱"
        }
    ]
    
    decision = random.choice(events)
    # 模拟判定后的自动回落逻辑（异步模式核心）
    time.sleep(3) 
    core.is_vision_active = False
    return jsonify(decision)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)