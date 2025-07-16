**冗余和不必要的代码标记（含行号）**

以下标注了源文件 `tooth_template_builder.py` 中的关键重复或冗余代码位置，行号基于当前文件版：

---

### 1. `choose_separation_method` 多次分支重复调用（文件行 671-678）

````python
671  # 智能选择高性能分离方法
672  if shape_complexity > 80 or convexity < 0.7:
673      return ultra_separate_connected_objects(mask)
674  elif compactness < 0.3:
675      return ultra_separate_connected_objects(mask)
676  else:
677      return advanced_separate_connected_objects(mask)
678  ```
- **问题**: 两个条件分支调用同一方法。
- **建议**: 合并为：
  ```python
672  if shape_complexity > 80 or convexity < 0.7 or compactness < 0.3:
673      return ultra_separate_connected_objects(mask)
674  return advanced_separate_connected_objects(mask)
````

---

### 2. 未使用的备用实现放在三引号中（文件行 610-640）

```python
606 def ultra_separate_connected_objects(mask):
607     # 超强黏连分离算法 - 仅使用OpenCV，无需额外依赖
...
638     return best_result
639     """
640     超强黏连分离算法 - 针对牙齿模型优化
...
```

* **问题**: 第 639-640 行后面三引号内的备选实现永远不会执行。
* **建议**: 删除第 639 行起的三引号及其内部内容。

---

### 3. 相似腐蚀-膨胀逻辑多处重复（文件行 643-705、713-745、755-785）

* `force_separation_with_morphology`（643-701）

* `advanced_separate_connected_objects`（713-740）

* `erosion_dilation_separation`（755-785）

* **问题**: 三个函数都实现了类似的腐蚀->连通分量检测->恢复->合并。

* **建议**: 提炼通用函数 `separate_by_morphology(mask, erosion_kernels, dilation_kernels)` 放行 643-785 之间的核心步骤，仅在高层策略中调用。

---

### 4. 模板列表功能重复（文件行 118-145 与 217-233）

* `list_templates`（118-145）通过数据库查询列出模板。

* `list_all_saved_templates`（217-233）通过文件系统扫描列出 JSON 文件。

* **建议**: 选用数据库查询即可，删除 `list_all_saved_templates` 或将其内部实现替换为对数据库的调用。

---

### 5. `auto_save` 分支几乎一致（文件行 100-112）

```python
100  save_type = "自动保存" if auto_save else "手动保存"
101  print(f"✅ 模板已{save_type}: {tooth_id} ({len(valid_contours)}个轮廓)")
```

* **问题**: 如果始终自动保存，可删除 `auto_save` 参数及相关条件。

---

### 6. 未按需导入依赖（文件行 3-10）

```python
3  import cv2
4  import numpy as np
5  import matplotlib.pyplot as plt
6  import matplotlib
7  import json
8  import sqlite3
9  import os
10 from datetime import datetime
11 from pathlib import Path
12 # 高性能库导入
13 from scipy import ndimage
14 from skimage.segmentation import watershed
```

* \*\*问题
