# Project 4: Build "Deep Research" Capability with Web Search and Reasoning Models - Textbook-Level Lecture Notes

[← กลับสู่หน้าหลัก (README.md)](../README.md)

---

## 1. Reasoning and Thinking LLMs (โมเดลภาษาแห่งการใช้เหตุผลเชิงลึก)

โมเดลยุคใหม่เช่น **OpenAI o1/o3** และ **DeepSeek-R1** เน้นการทำงานแบบ
**[System 2 Thinking](../glossary/system1_system2.md)**
ที่ใช้เวลาระดับหนึ่งในการจำลองภาพรวม ลบล้างข้อบกพร่อง และเรียบเรียงความคิดก่อนตอบ

### 1.1 ทำไม RL จึงสร้างพฤติกรรมคิดทบทวนตัวเองได้?

จากผลการทดลองของ DeepSeek-R1-Zero แสดงให้เห็นว่าการทำ **Reinforcement Learning
(RL)** ในระดับดิบโดยไม่มีข้อมูล SFT จากมนุษย์นำทาง
สามารถทำให้โมเดลพัฒนาพฤติกรรมการคิดทบทวนตัวเอง (Self-Correction)
และการแก้ไขปัญหาทีละเป้าหมายได้โดยอัตโนมัติ

ความสามารถนี้เกิดจากการที่ระบบให้รางวัล (Reward) แก่คำตอบสุดท้ายที่ถูกต้อง
และกำหนดบทลงโทษในกรณีที่โมเดลส่งผลลัพธ์ผิดพลาด
ส่งผลให้ตัวโมเดลจำยอมต้องหาวิถีทางสร้างโทเคนการคิดที่มีการตรวจสอบความเข้ากันได้ภายในบริบทบ่อยขึ้นเพื่อหาข้อผิดพลาดก่อนแสดงประโยคสุดท้าย

---

## 2. Inference-time Compute Scaling (การขยายสเกลคำนวณ ณ เวลาประมวลผล)

การขยายประสิทธิภาพของโมเดลภาษาย้อนกลับสามารถทำได้โดยการใส่กำลังการคำนวณเพิ่มเติมในช่วงทดสอบ
(Test-time Compute) แทนที่จะเพิ่มขนาดพารามิเตอร์ของโมเดล (Training-time Compute)

```text
       ▲
       │          /  (Test-time Compute Scaling Curve)
       │         /  
Accuracy │        /   
       │  ─────/     
       │ 
       └──────────────────────────────────►
          Inference Tokens (Search depth / rollouts)
```

### 2.1 Inference-Time Compute Scaling Laws

สมมติว่ากำหนดทรัพยากรการคำนวณสะสมให้สำหรับหาคำตอบมีค่าเท่ากับ $N$ โทเคน
เราสามารถออกแบบอัลกอริทึมเพื่อให้ได้ระดับความถูกต้องสูงสุดดังนี้:

- **Best-of-N Sampling (Rejection Sampling)**:
  สุ่มสร้างเส้นทางการคิดและคำตอบอิสระจำนวน $N$ รูปแบบคู่ขนาน
  จากนั้นส่งทั้งหมดให้ตัวตรวจความสัมพันธ์ภายนอก (Verifier)
  หรือให้ความหนาแน่นความน่าจะเป็นประเมินคะแนนสะสม
  เพื่อเลือกเพียงคำตอบที่มีระดับคะแนนสูงสุดมาใช้ตอบ

> [!TIP]
> **Code Example:** ดูวิธีการเขียนโปรแกรมสุ่มหาคำตอบที่ดีที่สุดจาก $N$ เส้นทางด้วย Verifier
> ได้ที่ไฟล์
> [project4_best_of_n_sampling.py](../code/project4_best_of_n_sampling.py)
>
> ```python
> # คัดลอกตรรกะบางส่วน (ดูโค้ดเต็มในลิงก์ด้านบน)
> def best_of_n_sampling(prompt, N=5):
>     paths = mock_generate_reasoning_paths(prompt, N)
>     best_path = max(paths, key=lambda p: mock_verifier_model(p))
>     return best_path
> ```

- **Tree of Thoughts (ToT) Exploration**:
  จัดตั้งรูปแบบกิ่งทางเลือกเป็นโครงสร้างกราฟและทำการค้นหาโดยใช้ขั้นตอน
  [BFS](../glossary/bfs_dfs.md) หรือ [DFS](../glossary/bfs_dfs.md)
  ร่วมกับการคัดกรองคะแนนความน่าจะเป็น เพื่อตัดกิ่งความคิดที่ไม่มีอนาคต (Pruning)
  หรือกระโดดถอยหลังกลับไปตั้งต้นคิดใหม่ (Backtracking)

---

## 3. Training-Time Reasoning Techniques (เทคนิคการสร้างโมเดลคิดวิเคราะห์)

### 3.1 อัลกอริทึม STaR (Self-Taught Reasoner)

STaR (Zelikman et al., 2022) เป็นอัลกอริทึมประเภท
[Bootstrapping](../glossary/bootstrapping.md)
ที่พัฒนาความสามารถการคิดของโมเดลผ่านการทบทวนความล้มเหลว:

1. **การสังเคราะห์ความคิด (Generation)**: ป้อนโจทย์ปัญหา $x \in \mathcal{D}$
   ให้โมเดลสร้าง Chain-of-Thought $r$ และคำตอบสุดท้าย $y$
2. **การกรองคุณภาพ (Filtering)**: หากคำตอบ $y$ ตรงกับเฉลยจริง $y^*$ เราจะบันทึกคู่ข้อมูล
   $(x, r)$ เข้าสู่ฐานข้อมูลเทรนชุดใหม่
3. **การปรับปรุงกระบวนการผ่าน Rationalization (กรณีตอบผิด)**: หากคำตอบที่ได้ผิดพลาด
   ($y \ne y^*$) เราจะช่วยกระตุ้นโมเดลโดยป้อนคำตอบเฉลยจริง $y^*$ แนบไปในพร้อมต์คำสั่ง
   เพื่อขอให้โมเดลอธิบายวิถีความคิดที่ถูกต้องย้อนกลับมา (Rationalization)
   นำคำอธิบายความคิดย้อนกลับที่ช่วยคิดได้สำเร็จนี้ไปบันทึกเพิ่มในคลังฝึก
4. **Fine-Tuning**: นำข้อมูลผลเฉลยความคิดทั้งหมดที่รวบรวมมาทำการเทรนโมเดลผ่าน Supervised
   Fine-Tuning ในรอบใหม่เพื่อปรับปรุงระดับความคิดเริ่มต้น

---

### 3.2 Reward Modeling สำหรับ Reasoning: ORM vs PRM

ในกระบวนการฝึกฝนระบบคำนวณเหตุผล การออกแบบโมเดลรางวัลมีความสำคัญมาก:

- **Outcome-based Reward Model (ORM)**: ให้คะแนนรางวัลเป็นค่าไบนารีบนผลลัพธ์สุดท้าย $Y$
  เท่านั้น:
  $$R(X, Y) = \begin{cases} +1 & \text{if } Y = Y^* \\ -1 & \text{if } Y \ne Y^* \end{cases}$$
  _ข้อดี_: ตรวจทานง่ายโดยใช้คอมไพเลอร์โค้ดหรือการเช็คสตริงผลลัพธ์คำตอบเลข _ข้อเสีย_:
  อาจเกิดปัญหาการลัดเส้นทางความคิด (Reward hacking)
  โดยที่โมเดลใช้วิธีคิดที่ผิดพลาดแต่บังเอิญตอบเลขถูก
- **Process-based Reward Model (PRM)**: คำนวณให้ระดับคะแนนรางวัลกับทุก ๆ
  ตำแหน่งการเปลี่ยนผ่านความคิดในแต่ละบรรทัด $y_t$:
  $$R(X, Y) = \sum_{t=1}^T r_\phi(x, y_{\le t})$$
  _ข้อดี_: ช่วยส่งเสริมลอจิกความคิดในระดับวิชาการ ลดปัญหาโมเดลโกงความคิด _ข้อเสีย_:
  สิ้นเปลืองข้อมูลในการทำ Annotation ระบุเกรดความคุ้มค่าแต่ละบรรทัดอย่างมหาศาล

---

## 4. Local Deployment and Quantization Mechanics

เพื่อการรันโมเดลขนาดใหญ่ภายในเครื่องที่มีข้อจำกัดทางกายภาพ AI Engineer
ต้องเข้าใจโครงสร้างการบีบอัดตัวแบบ

### 4.1 คณิตศาสตร์ของการทำ Quantization

การทำ Quantization แปลงตัวแปรทศนิยม FP32/FP16 ไปเป็นเลขจำนวนเต็มความละเอียดต่ำ (เช่น
INT8) ทำงานโดยการสเกลและการชดเชยค่าศูนย์
([Affine Quantization](../glossary/affine_quantization.md)):

$$q = \text{round}\left(\frac{x}{S}\right) + Z$$

โดยที่:

- $x \in [\beta, \alpha]$ คือช่วงตัวเลขของพารามิเตอร์ทศนิยมเดิม
- $q$ คือค่าจำลองจำนวนเต็มที่ได้หลังจาก Quantize (เช่น ช่วง $[-128, 127]$ สำหรับ INT8)
- $S$ คือค่าคงที่จัดสเกล (Scaling Factor):
  $$S = \frac{\alpha - \beta}{q_{\max} - q_{\min}}$$
- $Z$ คือจุดเริ่มศูนย์ (Zero-point) เพื่อแมปค่าศู่นย์เดิมให้สอดคล้องกับเลขจำนวนเต็มใหม่:
  $$Z = \text{round}\left(\frac{-\beta}{S}\right) + q_{\min}$$

การทำคืนรูป (Dequantization)
สำหรับนำค่าน้ำหนักไปคำนวณทางคณิตศาสตร์ต่อในตัวโมเดลจะกระทำดังนี้:

$$\tilde{x} = S \cdot (q - Z)$$

> [!TIP]
> **Code Example:** สัมผัสคณิตศาสตร์ของการแปลง FP32 เป็น INT8 พร้อมการทำ
> Dequantization ได้ที่ไฟล์
> [project4_affine_quantization.py](../code/project4_affine_quantization.py)
>
> ```python
> # ตัวอย่างการทำ Quantization (ดูโค้ดเต็มในลิงก์ด้านบน)
> def quantize_fp32_to_int8(weights_fp32):
>     S = (alpha - beta) / (q_max - q_min)
>     Z = np.round(-beta / S) + q_min
>     q = np.round(weights_fp32 / S) + Z
>     return np.clip(q, q_min, q_max).astype(np.int8)
> ```

---

### 4.2 vLLM PagedAttention Mechanics (การจัดการ KV Cache ขั้นสูง)

ในการทำเซิร์ฟเวอร์โมเดลภาษาขนาดใหญ่ ปัญหาคอขวดที่กินพื้นที่แรมบน GPU ที่สุดคือ KV Cache
ที่มีความยาวไม่แน่นอนและเกิดการจองเนื้อหาแบบกระจัดกระจาย (Fragmentation)

PagedAttention (Kwon et al., 2023) นำแนวคิดหน่วยความจำแบบหน้า
([Virtual Memory Paging](../glossary/virtual_memory_paging.md))
ในระบบปฏิบัติการมาประยุกต์ใช้กับ KV Cache:

```text
[Logical KV Cache Blocks] ──> Block Table ──> [Physical GPU Blocks (Non-contiguous)]
   Block 0 (Token 1-4)                          Physical Block 73
   Block 1 (Token 5-8)                          Physical Block 12
```

- **ลอจิกการทำงาน**: แทนที่จะจองพื้นที่แรม GPU แบบเรียงแผ่นยาวต่อเนื่องกันสำหรับโมเดลแต่ละเซสชัน
  PagedAttention จะแบ่ง KV Cache ออกเป็นบล็อกย่อยที่มีขนาดโทเคนคงที่ (เช่น 16 โทเคนต่อ 1
  บล็อก)
- **Block Table**: ระบบสร้างตารางแมปบล็อคคอยจับคู่ว่าบล็อคความคิดตำแหน่งตรรกะ (Logical
  Block) จะไปเชื่อมโยงกับแรม GPU บล็อกใด (Physical Block) ที่อยู่กระจัดกระจายแบบไม่ต่อเนื่อง
- วิธีการนี้ช่วยกำจัดพื้นที่สูญเสียในหน่วยความจำ GPU (Memory waste) ได้เกือบ 100%
  ส่งผลให้ระบบสามารถรองรับผู้ใช้งานพร้อมกัน (Throughput) ได้สูงขึ้น 2 ถึง 4 เท่าตัว

---

[← กลับสู่หน้าหลัก (README.md)](../README.md)
