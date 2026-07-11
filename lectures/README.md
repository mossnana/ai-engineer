# คลังเอกสารการบรรยายหลักสูตร AI Engineer (Course Lecture Notes)

ยินดีต้อนรับสู่คลังเอกสารการบรรยายหลักสูตรการสร้างวิศวกรปัญญาประดิษฐ์เชิงปฏิบัติการ (AI Engineer
Course by Projects) ซึ่งจัดทำสรุปเนื้อหาทฤษฎี ข้อกำหนด
และแนวทางปฏิบัติไว้อย่างละเอียดเชิงลึกระดับมหาวิทยาลัย (University Lecture Notes)
เป็นภาษาไทย

กรุณาเลือกหัวข้อที่ต้องการศึกษาตามรายการโปรเจกต์ด้านล่างนี้:

---

## 📚 สารบัญบทเรียนในแต่ละหัวข้อ

### 1. [Project 1: Build an LLM Playground](project1_llm_foundations.md)

- **คำอธิบายย่อย**: รากฐานโครงข่ายประสาทเทียม, สถาปัตยกรรม Transformers
  (Self-Attention), ความแตกต่าง GPT/DeepSeek/Qwen/Gemma, กระบวนการ Pre-training,
  Tokenization, Decoding Algorithms (Top-k, Top-p, Temp), Post-training (SFT,
  RLHF, PPO, GRPO), การวัดผล (PPL, Benchmarks, Arena) และการดีไซน์ระบบแชตบอต

### 2. [Project 2: Build a Customer Support Chatbot using RAGs and Prompt Engineering](project2_rag_prompt_engineering.md)

- **คำอธิบายย่อย**: การเปรียบเทียบเทคนิค Model Adaptation, ทฤษฎี PEFT และ LoRA
  (Low-Rank Adaptation), หลักสูตรพร้อมต์และการทำ Chain-of-Thought, สถาปัตยกรรม RAG,
  กระบวนการทำ Document Parsing และ Chunking, ดัชนีและการค้นหาเวกเตอร์ (Exact vs ANN,
  HNSW), เทคนิคการเทรน RAFT และกรอบประเมินผล RAG Triad

### 3. [Project 3: Build an "Ask-the-Web" Agent similar to Perplexity with Tool calling](project3_agentic_workflows_tools.md)

- **คำอธิบายย่อย**: ความแตกต่างระหว่าง LLM, Agentic System และ Agent, ดีไซน์เวิร์กโฟลว์
  (Prompt Chaining, Routing, Parallelization, Reflection, Orchestration),
  กลไกการยิงเรียก Tool calling, มาตรฐานโปรโตคอลเปิด MCP, เอเจนต์หลายระดับขั้นตอน
  (ReACT, ReWOO, Tree Search / MCTS), ปัญหาของระบบ Multi-Agent
  และการทดสอบวัดผลระดับอุตสาหกรรม (WebArena, SWE-bench)

### 4. [Project 4: Build "Deep Research" Capability with Web Search and Reasoning Models](project4_deep_research_reasoning.md)

- **คำอธิบายย่อย**: ระบบการคิดคำนวณขั้นสูง (System 2 Thinking), โมเดลสายเหตุผลตระกูล
  OpenAI o และ DeepSeek-R1, เทคนิค Inference-time Scaling (Parallel/Sequential
  sampling, ToT, Verifiers), การฝึกฝนโมเดลเพื่อใช้เหตุผล (STaR, RL on verifiable
  tasks, ORM vs PRM, Internalized CoT), และการติดตั้งใช้ในคอมพิวเตอร์ส่วนบุคคล
  (Quantization GGUF/AWQ, Ollama, vLLM)

### 5. [Project 5: Build a Multi-modal Generation Agent](project5_multimodal_generation.md)

- **คำอธิบายย่อย**: ภาพรวมโมเดลสร้างภาพและสื่อเคลื่อนไหว (VAE, GAN, AR, Diffusion),
  สถาปัตยกรรมสร้างรูปภาพจากคำสั่ง Text-to-Image (U-Net, DiT), สมการคณิตศาสตร์ขั้นตอน
  Forward และ Backward Diffusion, เทคนิคการสุ่มยุคใหม่ (DDPM, DDIM, Flow Matching),
  การวัดผลเชิงทัศนศิลป์ (FID, IS, CLIP Score) และระบบสร้างสรรค์วิดีโออสังหาริมทรัพย์ระดับลึก
  (3D Space-Time Attention, Latent Caching)

### 6. [Project 6: Capstone Project - Guidelines](project6_capstone_guide.md)

- **คำอธิบายย่อย**: กรอบการเริ่มโครงงานหลักสูตรเพื่อจบการศึกษา,
  เกณฑ์การตั้งโจทย์ที่มีความท้าทายทางวิศวกรรม, วิธีวิเคราะห์และวางโครงสร้างสถาปัตยกรรมระบบ,
  การออกแบบชุดตรวจสอบวัดผลคุณภาพโปรเจกต์, และกลวิธีเตรียม Git Repository, Portfolio
  และเตรียมความพร้อมสำหรับ Demo Day

---

[← กลับสู่หน้าหลักของคอร์ส (README.md)](../README.md)
