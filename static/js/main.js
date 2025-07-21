
function setupDragAndDrop(elementId, callback) {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    element.addEventListener('dragover', function(e) {
        e.preventDefault();
        element.classList.add('dragover');
    });
    
    element.addEventListener('dragleave', function(e) {
        e.preventDefault();
        element.classList.remove('dragover');
    });
    
    element.addEventListener('drop', function(e) {
        e.preventDefault();
        element.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0 && callback) {
            callback(files);
        }
    });
}

function showLoading(elementId, message = '加载中...') {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = `
            <div class="text-center">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">${message}</span>
                </div>
                <p class="mt-2">${message}</p>
            </div>
        `;
    }
}

function showError(elementId, message) {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-circle"></i> ${message}
            </div>
        `;
    }
}

function showSuccess(elementId, message) {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = `
            <div class="alert alert-success">
                <i class="fas fa-check-circle"></i> ${message}
            </div>
        `;
    }
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function validateImageFile(file) {
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/bmp'];
    const maxSize = 10 * 1024 * 1024; // 10MB
    
    if (!validTypes.includes(file.type)) {
        return { valid: false, message: '不支持的文件格式，请选择 PNG, JPG, JPEG 或 BMP 文件' };
    }
    
    if (file.size > maxSize) {
        return { valid: false, message: '文件大小超过限制（最大10MB）' };
    }
    
    return { valid: true };
}
