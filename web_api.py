from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import cv2
import numpy as np
import json
import os
from pathlib import Path
from typing import List, Dict, Optional
import tempfile
import shutil
from batch_processor import BatchTemplateProcessor
from cross_domain_config import CrossDomainConfig
from match import ToothMatcher, ContourFeatureExtractor
from BulidTheLab import ToothTemplateBuilder

app = FastAPI(title="TOOTHLAB 牙齿识别系统", version="1.0.0")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

batch_processor = BatchTemplateProcessor()
tooth_matcher = ToothMatcher()
template_builder = ToothTemplateBuilder()

class WebToothMatcher:
    def __init__(self):
        self.feature_extractor = ContourFeatureExtractor()
        self.matcher = ToothMatcher()
    
    def process_and_match(self, image_path, config_type='cross_domain'):
        """处理图像并执行匹配，支持跨域匹配"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return {"success": False, "message": "图像读取失败"}
            
            try:
                from cross_domain_config import CrossDomainConfig
                if config_type == 'modeling':
                    config = CrossDomainConfig.MODELING_IMAGE_CONFIG
                elif config_type == 'real_photo':
                    config = CrossDomainConfig.REAL_PHOTO_CONFIG
                else:
                    config = CrossDomainConfig.CROSS_DOMAIN_CONFIG
            except ImportError:
                config = {
                    'hsv_range': {'lower': [0, 0, 0], 'upper': [180, 255, 100]},
                    'thresholds': {'fine_match': 0.3}
                }
            
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv_range = config.get('hsv_range', {'lower': [0, 0, 0], 'upper': [180, 255, 100]})
            lower = np.array(hsv_range['lower'])
            upper = np.array(hsv_range['upper'])
            
            mask = cv2.inRange(hsv, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            query_features = []
            for contour in contours:
                if contour.shape[0] < 20 or cv2.contourArea(contour) < 100:
                    continue
                
                points = contour.reshape(-1, 2)
                features = self.feature_extractor.extract_all_features(contour, points, img.shape)
                query_features.append(features)
            
            if not query_features:
                return {"success": False, "message": "未检测到有效轮廓"}
            
            threshold = config.get('thresholds', {}).get('fine_match', 0.3)
            matches = self.matcher.match_against_database(query_features, threshold=threshold)
            
            results = []
            for query_id, query_matches in matches.items():
                for match in query_matches[:3]:
                    similarity_percent = match['similarity']
                    confidence = "高" if similarity_percent > 0.7 else "中等" if similarity_percent > 0.4 else "低"
                    
                    results.append({
                        "template_id": match['template_id'],
                        "similarity": similarity_percent,
                        "confidence": confidence,
                        "instruction": f"操作指令：{match['template_id']}"
                    })
            
            return {
                "success": True,
                "results": results,
                "config_used": config_type,
                "contours_found": len(query_features)
            }
            
        except Exception as e:
            return {"success": False, "message": f"处理失败: {str(e)}"}

web_matcher = WebToothMatcher()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """主页"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/templates", response_class=HTMLResponse)
async def template_management(request: Request):
    """模板管理界面"""
    templates_list = template_builder.list_all_saved_templates()
    return templates.TemplateResponse("templates.html", {
        "request": request,
        "templates": templates_list
    })

@app.get("/recognition", response_class=HTMLResponse)
async def recognition_interface(request: Request):
    """识别界面"""
    return templates.TemplateResponse("recognition.html", {"request": request})

@app.post("/api/batch_upload")
async def batch_upload_templates(files: List[UploadFile] = File(...)):
    """批量上传模板图像"""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            for file in files:
                if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    continue
                    
                file_path = Path(temp_dir) / file.filename
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
            
            results = batch_processor.process_directory(temp_dir, auto_confirm=True)
            
        return JSONResponse({
            "success": True,
            "message": f"成功处理 {results['processed']} 个文件",
            "results": results
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/recognize")
async def recognize_tooth(file: UploadFile = File(...)):
    """识别单个牙齿图像"""
    try:
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            raise HTTPException(status_code=400, detail="不支持的文件格式")
        
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="无法解析图像文件")
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            cv2.imwrite(temp_file.name, img)
            
            try:
                results = web_matcher.process_and_match(temp_file.name)
                return JSONResponse({
                    "success": True,
                    "results": results
                })
            finally:
                os.unlink(temp_file.name)
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/templates")
async def get_templates():
    """获取所有模板列表"""
    try:
        templates_list = template_builder.list_all_saved_templates()
        return JSONResponse({
            "success": True,
            "templates": templates_list
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/templates/{template_id}")
async def delete_template(template_id: str):
    """删除指定模板"""
    try:
        success = template_builder.delete_template(template_id)
        if success:
            return JSONResponse({
                "success": True,
                "message": f"模板 {template_id} 已删除"
            })
        else:
            raise HTTPException(status_code=404, detail="模板不存在")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_system_stats():
    """获取系统统计信息"""
    try:
        templates_list = template_builder.list_all_saved_templates()
        templates_count = len(templates_list) if templates_list else 0
        return JSONResponse({
            "success": True,
            "stats": {
                "total_templates": templates_count,
                "system_status": "运行正常"
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
