from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, JSONResponse
import io
from RegionalDiffusion_xl import RegionalDiffusionXLPipeline
from diffusers.schedulers import DPMSolverMultistepScheduler
from diffusers import DiffusionPipeline
from mllm import GPT5
import torch
from pydantic import BaseModel

gpt_token = "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"
hf_cache_dir = "/workspace/my_models/cache"

app = FastAPI()

negative_prompt = "worst quality, low quality, medium quality, deleted, lowres, comic, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, jpeg artifacts, signature, watermark, username, blurry"

class PromptRequest(BaseModel):
    split_ratio: str
    regional_prompt: str
    original_prompt: str
    prompt_en: str

# --- 1. API Trả về giao diện HTML ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_content = """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RPG-DiffusionMaster</title>
        <style>
            body { font-family: 'Segoe UI', sans-serif; padding: 20px; max-width: 1500px; margin: 0 auto; }
            .control-group { margin-bottom: 20px; display: flex; gap: 10px; }
            input { flex: 1; padding: 10px; font-size: 16px; }
            button { padding: 10px 20px; font-size: 16px; cursor: pointer; background: #007bff; color: white; border: none; border-radius: 4px; }
            button:disabled { background: #ccc; cursor: not-allowed; }
            
            label { font-weight: bold; display: block; margin-top: 10px; }
            textarea { width: 100%; height: 80px; padding: 10px; margin-bottom: 20px; font-family: monospace; border: 1px solid #ccc; border-radius: 4px; }
            
            .image-container { display: flex; gap: 15px; justify-content: space-between; margin-top: 10px; }
            .image-box { flex: 1; text-align: center; border: 1px solid #ddd; padding: 10px; border-radius: 8px; background: #f9f9f9; }
            .image-box img { width: 100%; height: auto; display: block; margin-top: 10px; border-radius: 4px; }
            .image-box h3 { margin: 0; color: #333; font-size: 14px; }
            
            .loading { color: #666; font-style: italic; display: none; }
        </style>
    </head>
    <body>
        <h1>RPG-DiffusionMaster Demo</h1>
        
        <div class="control-group">
            <input type="text" id="rawPrompt" placeholder="Nhập ý tưởng ...">
            <button id="btnGen" onclick="runWorkflow()">Generate</button>
        </div>

        <label>Prompt đã xử lý:</label>
        <textarea id="refinedPrompt" readonly></textarea>
        <div class="image-container">
            <div class="image-box">
                <h3>SDXL-Turbo</h3>
                <p id="loading-sdxl-legacy" class="loading">Đang vẽ...</p>
                <img id="img-sdxl-legacy" />
            </div>
            <div class="image-box">
                <h3>Playground v2</h3>
                <p id="loading-playg-legacy" class="loading">Đang vẽ...</p>
                <img id="img-playg-legacy" />
            </div>
            <div class="image-box">
                <h3>IterComp (RPG included while training)</h3>
                <p id="loading-iter-legacy" class="loading">Đang vẽ...</p>
                <img id="img-iter-legacy" />
            </div>
        </div>
        <div class="image-container">
            <div class="image-box">
                <h3>SDXL-Turbo + RPG</h3>
                <p id="loading-sdxl" class="loading">Đang vẽ...</p>
                <img id="img-sdxl" />
            </div>
            <div class="image-box">
                <h3>Playground v2 + RPG</h3>
                <p id="loading-playg" class="loading">Đang vẽ...</p>
                <img id="img-playg" />
            </div>
            <div class="image-box">
                <h3>IterComp + RPG</h3>
                <p id="loading-iter" class="loading">Đang vẽ...</p>
                <img id="img-iter" />
            </div>
        </div>

        <script>
            // Hàm dùng chung để gọi API và cập nhật giao diện của RIÊNG box đó
            async function fetchAndShow(url, payload, imgId, loadingId) {
                const imgEl = document.getElementById(imgId);
                const loadingEl = document.getElementById(loadingId);
                
                try {
                    const res = await fetch(url, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });

                    if (!res.ok) throw new Error("API Error");

                    const blob = await res.blob();
                    const imgUrl = URL.createObjectURL(blob);
                    
                    // Cập nhật ngay khi API này xong (không chờ API kia)
                    loadingEl.style.display = 'none';
                    imgEl.src = imgUrl;
                    imgEl.style.display = 'block';
                    
                } catch (error) {
                    loadingEl.innerText = "Lỗi!";
                    console.error(error);
                }
            }

            async function runWorkflow() {
                const rawText = document.getElementById('rawPrompt').value;
                const refinedArea = document.getElementById('refinedPrompt');
                const btn = document.getElementById('btnGen');
                
                // 1. Reset trạng thái
                btn.disabled = true;
                refinedArea.value = "Processing...";
                
                // Ẩn ảnh cũ, hiện loading
                ['sdxl', 'playg', 'sdxl-legacy', 'playg-legacy', 'iter', 'iter-legacy'].forEach(type => {
                    document.getElementById(`img-${type}`).style.display = 'none';
                    //document.getElementById(`loading-${type}`).style.display = 'block';
                    //document.getElementById(`loading-${type}`).innerText = 'Đang vẽ...';
                });

                try {
                    // 2. Gọi API Prompt
                    const promptRes = await fetch(`/api/prompt?text=${encodeURIComponent(rawText)}`);
                    const promptJson = await promptRes.json();
                    const finalPrompt = promptJson.regional_prompt;
                    refinedArea.value = finalPrompt;

                    //const payload = { text: finalPrompt };

                    // 3. Kích hoạt 2 request SONG SONG và ĐỘC LẬP
                    // Lưu ý: Ta không dùng "await" ở trước mỗi hàm fetchAndShow riêng lẻ
                    // mà gán nó vào biến Promise

                    // Ẩn ảnh cũ, hiện loading
                    ['sdxl', 'playg', 'sdxl-legacy', 'playg-legacy', 'iter', 'iter-legacy'].forEach(type => {
                        //document.getElementById(`img-${type}`).style.display = 'none';
                        document.getElementById(`loading-${type}`).style.display = 'block';
                        document.getElementById(`loading-${type}`).innerText = 'Đang vẽ...';
                    });

                    const taskSDXLLegacy = await fetchAndShow('/api/SDXLLegacy', promptJson, 'img-sdxl-legacy', 'loading-sdxl-legacy');
                    const taskPlaygLegacy = await fetchAndShow('/api/PlaygLegacy', promptJson, 'img-playg-legacy', 'loading-playg-legacy');
                    const taskIterLegacy = await fetchAndShow('/api/IterCompLegacy', promptJson, 'img-iter-legacy', 'loading-iter-legacy');
                    
                    const taskSDXL = await fetchAndShow('/api/SDXL', promptJson, 'img-sdxl', 'loading-sdxl');
                    const taskPlayg = await fetchAndShow('/api/Playg', promptJson, 'img-playg', 'loading-playg');
                    const taskIter = await fetchAndShow('/api/IterComp', promptJson, 'img-iter', 'loading-iter');

                    // 4. Chờ cả 2 xong chỉ để bật lại nút Button (Logic UI tổng)
                    // Việc hiển thị ảnh đã được xử lý bên trong từng task rồi.
                    //await Promise.all([taskSDXLLegacy, taskPlaygLegacy, taskSDXL, taskPlayg, taskIter, taskIterLegacy]);

                } catch (error) {
                    alert("Có lỗi: " + error.message);
                } finally {
                    btn.disabled = false;
                }
            }
     
        </script>
    </body>
    </html>
    """
    return html_content

# --- API 1: Xử lý Prompt (Text -> Text) ---
@app.get("/api/prompt")
async def process_prompt(text: str = ""):
    para_dict = GPT5(text, gpt_token)
    return JSONResponse(content={'original_prompt': text, 'regional_prompt': para_dict['Regional Prompt'], 
                                 'split_ratio': para_dict['Final split ratio'], 'prompt_en': para_dict['Prompt En']})

@app.post("/api/IterComp")
async def generate_image1(promptData: PromptRequest):

    pipe1 = RegionalDiffusionXLPipeline.from_pretrained("comin/IterComp", cache_dir=hf_cache_dir, torch_dtype=torch.float16, use_safetensors=True)
    pipe1.to("cuda")
    pipe1.scheduler = DPMSolverMultistepScheduler.from_config(pipe1.scheduler.config,use_karras_sigmas=True)
    pipe1.enable_xformers_memory_efficient_attention()

    images = pipe1(
        prompt = promptData.regional_prompt,
        split_ratio = promptData.split_ratio, # The ratio of the regional prompt, the number of prompts is the same as the number of regions
        batch_size = 1,
        base_ratio = 0.6, # The ratio of the base prompt    
        base_prompt= promptData.prompt_en,       
        num_inference_steps=50, # sampling step
        height = 800, 
        negative_prompt=negative_prompt, # negative prompt
        width = 800, 
        seed = 48, # random seed
        guidance_scale = 7.0
    ).images[0]
    # images.save("test2.png")

    img_io = io.BytesIO()
    images.save(img_io, 'PNG')
    
    # FastAPI trả về trực tiếp các bytes với định dạng image/png
    return Response(content=img_io.getvalue(), media_type="image/png")
    
@app.post("/api/Playg")
async def generate_image2(promptData: PromptRequest):
    pipe2 = RegionalDiffusionXLPipeline.from_pretrained("playgroundai/playground-v2-512px-base", cache_dir=hf_cache_dir, torch_dtype=torch.float16, use_safetensors=True)
    pipe2.to("cuda")
    pipe2.scheduler = DPMSolverMultistepScheduler.from_config(pipe2.scheduler.config,use_karras_sigmas=True)
    pipe2.enable_xformers_memory_efficient_attention()

    images = pipe2(
        prompt = promptData.regional_prompt,
        split_ratio = promptData.split_ratio, # The ratio of the regional prompt, the number of prompts is the same as the number of regions
        batch_size = 1,
        base_ratio = 0.6, # The ratio of the base prompt    
        base_prompt= promptData.prompt_en,       
        num_inference_steps=50, # sampling step
        height = 800, 
        negative_prompt=negative_prompt, # negative prompt
        width = 800, 
        seed = 48, # random seed
        guidance_scale = 7.0
    ).images[0]
    # images.save("test2.png")

    img_io = io.BytesIO()
    images.save(img_io, 'PNG')
    
    # FastAPI trả về trực tiếp các bytes với định dạng image/png
    return Response(content=img_io.getvalue(), media_type="image/png")

@app.post("/api/SDXL")
async def generate_image3(promptData: PromptRequest):
    pipe3 = RegionalDiffusionXLPipeline.from_pretrained("stabilityai/sdxl-turbo", cache_dir=hf_cache_dir, torch_dtype=torch.float16, use_safetensors=True)
    pipe3.to("cuda")
    pipe3.scheduler = DPMSolverMultistepScheduler.from_config(pipe3.scheduler.config,use_karras_sigmas=True)
    pipe3.enable_xformers_memory_efficient_attention()

    images = pipe3(
        prompt = promptData.regional_prompt,
        split_ratio = promptData.split_ratio, # The ratio of the regional prompt, the number of prompts is the same as the number of regions
        batch_size = 1,
        base_ratio = 0.6, # The ratio of the base prompt    
        base_prompt= promptData.prompt_en,       
        num_inference_steps=50, # sampling step
        height = 800, 
        negative_prompt=negative_prompt, # negative prompt
        width = 800, 
        seed = 48, # random seed
        guidance_scale = 7.0
    ).images[0]
    # images.save("test2.png")

    img_io = io.BytesIO()
    images.save(img_io, 'PNG')
    
    # FastAPI trả về trực tiếp các bytes với định dạng image/png
    return Response(content=img_io.getvalue(), media_type="image/png")

@app.post("/api/PlaygLegacy")
async def generate_image4(promptData: PromptRequest):
    pipe4 = DiffusionPipeline.from_pretrained("playgroundai/playground-v2-512px-base", cache_dir=hf_cache_dir, torch_dtype=torch.float16, use_safetensors=True)
    pipe4.to("cuda")

    images = pipe4(prompt=promptData.prompt_en,width = 800,height = 800).images[0]
    img_io = io.BytesIO()
    images.save(img_io, 'PNG')
    
    # FastAPI trả về trực tiếp các bytes với định dạng image/png
    return Response(content=img_io.getvalue(), media_type="image/png")
    
@app.post("/api/SDXLLegacy")
async def generate_image5(promptData: PromptRequest):
    pipe5 = DiffusionPipeline.from_pretrained("stabilityai/sdxl-turbo", cache_dir=hf_cache_dir, torch_dtype=torch.float16, use_safetensors=True)
    pipe5.to("cuda")

    images = pipe5(prompt=promptData.prompt_en,width = 800,height = 800).images[0]
    img_io = io.BytesIO()
    images.save(img_io, 'PNG')
    
    # FastAPI trả về trực tiếp các bytes với định dạng image/png
    return Response(content=img_io.getvalue(), media_type="image/png")

@app.post("/api/IterCompLegacy")
async def generate_image6(promptData: PromptRequest):
    pipe6 = DiffusionPipeline.from_pretrained("comin/IterComp", cache_dir=hf_cache_dir, torch_dtype=torch.float16, use_safetensors=True)
    pipe6.to("cuda")

    images = pipe6(prompt=promptData.prompt_en,width = 800,height = 800, num_inference_steps=50, base_ratio = 0.6).images[0]
    img_io = io.BytesIO()
    images.save(img_io, 'PNG')
 
    # FastAPI trả về trực tiếp các bytes với định dạng image/png
    return Response(content=img_io.getvalue(), media_type="image/png")

