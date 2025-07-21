# 云盘图片自我测试功能使用指南

## 功能概述

本功能允许用户从云盘下载牙齿图片，并通过手动确认的方式进行自我测试，验证TOOTHLAB系统的识别准确性。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 基本使用

```bash
# 单个URL测试
python run_self_test.py --urls "https://example.com/tooth1.jpg" --session-name "测试会话1"

# 多个URL批量测试
python run_self_test.py \
  --urls "https://example.com/tooth1.jpg,https://example.com/tooth2.jpg,https://example.com/tooth3.jpg" \
  --session-name "批量测试会话"

# 带预期结果的测试
python run_self_test.py \
  --urls "https://example.com/tooth1.jpg,https://example.com/tooth2.jpg" \
  --session-name "准确性测试" \
  --expected-templates "TOOTH_001,TOOTH_002"
```

### 3. 高级选项

```bash
# 清空下载缓存
python run_self_test.py --urls "..." --session-name "..." --clear-cache
```

## 工作流程

1. **下载阶段**: 系统从提供的URL下载图片到本地
2. **验证阶段**: 验证图片格式和完整性
3. **处理阶段**: 提取轮廓并与模板库匹配
4. **确认阶段**: 显示交互式界面供用户手动确认
5. **记录阶段**: 保存测试结果到数据库
6. **报告阶段**: 生成详细的测试报告

## 交互式确认界面

测试过程中会显示包含以下信息的界面：

- **测试图片**: 原始下载的图片
- **检测轮廓**: 系统检测到的牙齿轮廓
- **匹配结果**: 最佳匹配的模板和相似度
- **预期对比**: 如果提供了预期结果，会显示对比信息
- **处理统计**: 轮廓数量、模板库大小等统计信息
- **操作按钮**: 确认正确/确认错误/跳过

## 测试报告

每次测试完成后会生成JSON格式的详细报告，包含：

```json
{
  "session_id": 1,
  "session_name": "测试会话1",
  "created_at": "2025-07-21 06:22:05",
  "total_images": 3,
  "confirmed_count": 3,
  "correct_count": 2,
  "accuracy_rate": 0.67,
  "results": [
    {
      "image_name": "tooth1.jpg",
      "expected_template": "TOOTH_001",
      "matched_template": "TOOTH_001",
      "similarity_score": 0.92,
      "user_confirmed": true,
      "user_marked_correct": true,
      "processing_time": 1.5
    }
  ]
}
```

## 配置选项

可以通过修改 `config.py` 调整以下参数：

### 云盘下载配置
- `download_dir`: 下载目录
- `max_file_size`: 最大文件大小限制
- `timeout`: 下载超时时间
- `retry_times`: 重试次数

### 测试配置
- `similarity_threshold`: 相似度阈值
- `auto_confirm_threshold`: 自动确认阈值
- `batch_size`: 批处理大小

### UI配置
- `figure_size`: 界面窗口大小
- `font_size`: 字体大小
- `confirmation_timeout`: 确认超时时间

## 数据库结构

系统使用SQLite数据库存储测试结果：

### test_sessions 表
- `id`: 会话ID
- `session_name`: 会话名称
- `created_at`: 创建时间
- `total_images`: 总图片数
- `confirmed_matches`: 已确认数量
- `correct_matches`: 正确匹配数
- `accuracy_rate`: 准确率

### test_results 表
- `id`: 结果ID
- `session_id`: 所属会话ID
- `image_path`: 图片路径
- `expected_template_id`: 预期模板ID
- `matched_template_id`: 匹配到的模板ID
- `similarity_score`: 相似度分数
- `user_confirmed`: 用户是否确认
- `user_marked_correct`: 用户标记是否正确
- `processing_time`: 处理时间

## 故障排除

### 常见问题

1. **下载失败**
   - 检查URL是否有效
   - 确认网络连接正常
   - 验证图片格式是否支持

2. **模板库为空**
   - 使用 `BulidTheLab.py` 先创建模板库
   - 确认 `tooth_templates.db` 文件存在

3. **界面无响应**
   - 确认matplotlib后端配置正确
   - 检查是否在支持GUI的环境中运行

4. **依赖安装失败**
   - 更新pip: `pip install --upgrade pip`
   - 使用虚拟环境避免冲突

### 日志查看

系统会生成详细的日志文件 `self_test.log`，包含：
- 下载过程信息
- 图片处理详情
- 错误和警告信息
- 性能统计数据

## 扩展开发

### 添加新的云盘服务

在 `cloud_downloader.py` 中扩展 `CloudImageDownloader` 类：

```python
def download_from_baidu_cloud(self, share_url: str) -> Optional[Path]:
    # 实现百度云下载逻辑
    pass
```

### 自定义确认界面

继承 `SelfTester` 类并重写 `_show_test_confirmation_display` 方法：

```python
class CustomSelfTester(SelfTester):
    def _show_test_confirmation_display(self, ...):
        # 自定义界面逻辑
        pass
```

### 添加新的测试指标

在 `generate_test_report` 方法中添加自定义统计：

```python
def generate_test_report(self, session_id: int) -> Dict:
    report = super().generate_test_report(session_id)
    # 添加自定义指标
    report['custom_metrics'] = self.calculate_custom_metrics()
    return report
```

## 技术支持

如遇到问题，请检查：
1. 系统日志文件
2. 数据库内容
3. 模板库状态
4. 网络连接情况

更多技术细节请参考源代码注释和文档。
</new_str>
