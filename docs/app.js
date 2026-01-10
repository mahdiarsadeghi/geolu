/**
 * Geolu - Where Algorithms Predict Value
 * Copyright (c) 2026 Mahdiar Sadeghi
 * Licensed under Proprietary License with Educational Use
 * See LICENSE file for full terms
 */

// Global variables
let priceChart = null;

// Fetch and display data
async function loadData() {
    try {
        const response = await fetch('data.json');
        const data = await response.json();
        
        updateStatusBar(data);
        createPriceChart(data);
        displayPredictions(data);
    } catch (error) {
        console.error('Error loading data:', error);
        document.getElementById('predictions-grid').innerHTML = 
            '<div class="loading">Error loading data. Please run the predictor first.</div>';
    }
}

// Update status bar
function updateStatusBar(data) {
    document.getElementById('last-updated').textContent = formatDateTime(data.last_updated);
    document.getElementById('current-price').textContent = `$${data.current_price.toFixed(2)} USD/oz`;
    document.getElementById('target-date').textContent = formatDate(data.target_date);
}

// Create price chart with predictions
function createPriceChart(data) {
    const ctx = document.getElementById('priceChart').getContext('2d');
    
    // Destroy existing chart if it exists
    if (priceChart) {
        priceChart.destroy();
    }
    
    // Prepare historical data
    const historicalDates = data.historical.dates;
    const historicalPrices = data.historical.prices;
    
    // Get next week dates (Monday to Saturday)
    const targetDate = new Date(data.target_date);
    const futureDates = [];
    const currentDate = new Date(historicalDates[historicalDates.length - 1]);
    
    for (let i = 1; i <= 7; i++) {
        const nextDate = new Date(currentDate);
        nextDate.setDate(currentDate.getDate() + i);
        futureDates.push(nextDate.toISOString().split('T')[0]);
    }
    
    // Create datasets for each model
    const datasets = [
        {
            label: 'Historical Price',
            data: historicalPrices,
            borderColor: '#667eea',
            backgroundColor: 'rgba(102, 126, 234, 0.1)',
            borderWidth: 3,
            pointRadius: 2,
            pointHoverRadius: 5,
            tension: 0.4,
            fill: true
        }
    ];
    
    // Add prediction lines for each model
    const colors = {
        'Random Forest': '#10b981',
        'Linear Regression': '#f59e0b',
        'Ridge Regression': '#3b82f6',
        'Gradient Boosting': '#8b5cf6',
        'SVR': '#ec4899'
    };
    
    let colorIndex = 0;
    for (const [modelName, prediction] of Object.entries(data.predictions)) {
        const predictionLine = new Array(historicalPrices.length - 1).fill(null);
        predictionLine.push(historicalPrices[historicalPrices.length - 1]);
        predictionLine.push(prediction.predicted_price);
        
        datasets.push({
            label: `${modelName} Prediction`,
            data: predictionLine,
            borderColor: colors[modelName] || `hsl(${colorIndex * 60}, 70%, 50%)`,
            borderWidth: 2,
            borderDash: [5, 5],
            pointRadius: 4,
            pointHoverRadius: 6,
            tension: 0,
            fill: false
        });
        
        colorIndex++;
    }
    
    // Combine all dates
    const allDates = [...historicalDates, futureDates[0]];
    
    priceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: allDates,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 15
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed.y !== null) {
                                label += '$' + context.parsed.y.toFixed(2);
                            }
                            return label;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    ticks: {
                        callback: function(value) {
                            return '$' + value.toFixed(0);
                        }
                    }
                },
                x: {
                    ticks: {
                        maxRotation: 45,
                        minRotation: 45
                    }
                }
            }
        }
    });
}

// Display predictions
function displayPredictions(data) {
    const predictionsGrid = document.getElementById('predictions-grid');
    predictionsGrid.innerHTML = '';
    
    for (const [modelName, prediction] of Object.entries(data.predictions)) {
        const card = document.createElement('div');
        card.className = 'prediction-card';
        
        const changeClass = prediction.price_change_percent >= 0 ? 'positive' : 'negative';
        const changeSymbol = prediction.price_change_percent >= 0 ? '↑' : '↓';
        
        card.innerHTML = `
            <h3>${modelName}</h3>
            <div class="pred-price">$${prediction.predicted_price.toFixed(2)}</div>
            <div class="pred-change ${changeClass}">
                ${changeSymbol} ${Math.abs(prediction.price_change_percent).toFixed(2)}%
            </div>
        `;
        
        predictionsGrid.appendChild(card);
    }
}

// Helper functions
function formatDateTime(dateStr) {
    const date = new Date(dateStr);
    return date.toLocaleString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

function formatDate(dateStr) {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', {
        weekday: 'short',
        month: 'short',
        day: 'numeric',
        year: 'numeric'
    });
}

// Load data on page load
document.addEventListener('DOMContentLoaded', loadData);
