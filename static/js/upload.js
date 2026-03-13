// File selection display
document.getElementById('fileInput').addEventListener('change', function(e) {
    const fileName = e.target.files[0]?.name || 'Drag & Drop or Click to Choose CSV File';
    document.getElementById('fileName').textContent = '✅ ' + fileName;
});

// Drag and drop
const fileLabel = document.querySelector('.file-label');

fileLabel.addEventListener('dragover', (e) => {
    e.preventDefault();
    fileLabel.style.borderColor = '#764ba2';
    fileLabel.style.background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
    fileLabel.style.color = 'white';
});

fileLabel.addEventListener('dragleave', () => {
    fileLabel.style.borderColor = '#667eea';
    fileLabel.style.background = 'white';
    fileLabel.style.color = '#667eea';
});

fileLabel.addEventListener('drop', (e) => {
    e.preventDefault();
    const files = e.dataTransfer.files;
    document.getElementById('fileInput').files = files;
    document.getElementById('fileName').textContent = '✅ ' + files[0].name;
    fileLabel.style.borderColor = '#667eea';
    fileLabel.style.background = 'white';
    fileLabel.style.color = '#667eea';
});

// Form submit
document.getElementById('uploadForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('⚠️ Please select a CSV file!');
        return;
    }
    
    document.getElementById('loading').style.display = 'block';
    document.getElementById('results').style.display = 'none';
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            displayResults(data);
        } else {
            alert('❌ ' + data.error);
        }
    } catch (error) {
        alert('❌ ' + error.message);
    } finally {
        document.getElementById('loading').style.display = 'none';
    }
});

function displayResults(data) {
    const resultsDiv = document.getElementById('resultsContent');
    
    let html = `
        <div style="text-align: center; margin-bottom: 30px;">
            <div class="metric-box">
                <div class="metric-label">📝 Total Reviews</div>
                <div class="metric-value">${data.total_reviews}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">🎯 Accuracy</div>
                <div class="metric-value">${(data.accuracy * 100).toFixed(1)}%</div>
            </div>
        </div>
        
        <h3 style="margin: 40px 0 20px 0; color: #f57f17;">😊 Sentiment Distribution</h3>
        <table>
            <tr><th>Sentiment</th><th>Count</th><th>%</th></tr>
            ${Object.entries(data.sentiment_distribution).map(([key, val]) => {
                const emoji = key === 'positive' ? '😊' : key === 'negative' ? '😞' : '😐';
                const percent = ((val / data.total_reviews) * 100).toFixed(1);
                return `<tr><td>${emoji} ${key}</td><td>${val}</td><td>${percent}%</td></tr>`;
            }).join('')}
        </table>
        
        <h3 style="margin: 40px 0 20px 0; color: #f57f17;">🔑 Top Aspects</h3>
        <table>
            <tr><th>Aspect</th><th>😊 Positive</th><th>😐 Neutral</th><th>😞 Negative</th></tr>
            ${data.aspects.map(a => `
                <tr>
                    <td><strong>${a.aspect}</strong></td>
                    <td style="color: #4caf50;">${a.positive}</td>
                    <td style="color: #ff9800;">${a.neutral}</td>
                    <td style="color: #f44336;">${a.negative}</td>
                </tr>
            `).join('')}
        </table>
    `;
    
    resultsDiv.innerHTML = html;
    document.getElementById('results').style.display = 'block';
    document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
}

// Download sample CSV button
function downloadSample() {
    const csv = `review_text,rating
Great product! Really satisfied,5
Terrible experience,1
Average product,3
Excellent quality,5
Poor service,2
Good value,4
Disappointing,2
Amazing product,5
Okay delivery slow,3
Worst purchase,1
Best buy,5
Not bad,3
Loved it,5
Horrible quality,1
Decent price,4
Superb service,5
Waste of money,1
Fine product,3
Outstanding,5
Poor packaging,2
Fantastic,5
Bad quality,1
Pretty average,3
Excellent,5
Disappointed,2
Very good,4
Horrible,1
Neutral,3
Amazing,5
Below average,2`;

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'sample_reviews.csv';
    a.click();
}