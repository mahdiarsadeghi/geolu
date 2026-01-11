/**
 * Geolu - Where Algorithms Predict Value
 * Copyright (c) 2026 Mahdiar Sadeghi
 * Licensed under Proprietary License with Educational Use
 * See LICENSE file for full terms
 */

// Global variables
let historicalChart = null;
let predictionsChart = null;
let chartData = null;
let currentPeriod = 'weekly';

// Fetch and display data
async function loadData() {
    try {
        console.log('Loading data...');
        const response = await fetch('data.json');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        console.log('Data loaded successfully:', data);
        chartData = data;
        
        updateStatusBar(data);
        createHistoricalChart(data, currentPeriod);
        createPredictionsChart(data, currentPeriod);
        setupTabListeners();
    } catch (error) {
        console.error('Error loading data:', error);
    }
}

// Setup tab button listeners
function setupTabListeners() {
    const tabButtons = document.querySelectorAll('.tab-button');
    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Remove active class from all buttons
            tabButtons.forEach(btn => btn.classList.remove('active'));
            // Add active class to clicked button
            this.classList.add('active');
            // Update charts
            currentPeriod = this.getAttribute('data-period');
            createHistoricalChart(chartData, currentPeriod);
            createPredictionsChart(chartData, currentPeriod);
        });
    });
}

// Update status bar
function updateStatusBar(data) {
    document.getElementById('last-updated').textContent = formatDateTime(data.last_updated);
}

// Create historical prices chart for multiple assets
function createHistoricalChart(data, period = 'weekly') {
    console.log('Creating chart for period:', period);
    const canvas = document.getElementById('historicalChart');
    if (!canvas) {
        console.error('Canvas element not found');
        return;
    }
    
    const ctx = canvas.getContext('2d');
    if (!ctx) {
        console.error('Could not get canvas context');
        return;
    }
    
    // Destroy existing chart if it exists
    if (historicalChart) {
        historicalChart.destroy();
    }
    
    // Get data for the selected period
    let dataSource;
    if (data.time_periods && data.time_periods[period]) {
        dataSource = data.time_periods[period];
    } else if (period === 'weekly' && data.past_week) {
        dataSource = data.past_week;
    } else {
        dataSource = data.all_assets;
    }
    
    if (!dataSource) {
        console.error('No asset data available');
        return;
    }
    
    console.log('Data source:', dataSource);
    
    // Colors for each asset
    const colors = {
        'Gold': '#f59e0b',
        'Bitcoin': '#f97316',
        'Oil': '#0ea5e9',
        'S&P 500': '#8b5cf6'
    };
    
    // Create datasets for each asset with actual prices (not normalized)
    const datasets = [];
    const assetNames = Object.keys(dataSource);
    
    // Get common date range
    let allDates = [];
    for (const assetName of assetNames) {
        const asset = dataSource[assetName];
        if (asset.dates && asset.dates.length > 0) {
            allDates = asset.dates;
            break;
        }
    }
    
    // Create a dataset for each asset on its own y-axis
    for (const assetName of assetNames) {
        const asset = dataSource[assetName];
        if (!asset.prices || asset.prices.length === 0) continue;
        
        datasets.push({
            label: assetName,
            data: asset.prices,
            borderColor: colors[assetName] || '#667eea',
            backgroundColor: 'transparent',
            borderWidth: 2.5,
            pointRadius: 0,
            pointHoverRadius: 0,
            tension: 0.3,
            yAxisID: assetName.replace(/[^a-zA-Z0-9]/g, '') // Create unique y-axis ID
        });
    }
    
    // Create y-axes configuration for each asset
    const yAxes = {};
    assetNames.forEach((assetName, index) => {
        const axisId = assetName.replace(/[^a-zA-Z0-9]/g, '');
        const isRightSide = index % 2 === 1;
        yAxes[axisId] = {
            type: 'linear',
            display: true,
            position: isRightSide ? 'right' : 'left',
            grid: {
                drawOnChartArea: index === 0, // Only show grid for first axis
            },
            title: {
                display: true,
                text: assetName,
                color: colors[assetName] || '#667eea'
            },
            ticks: {
                color: colors[assetName] || '#667eea',
                callback: function(value) {
                    return '$' + value.toLocaleString();
                }
            }
        };
    });
    
    if (typeof Chart === 'undefined') {
        console.error('Chart.js library not loaded');
        return;
    }
    
    try {
        historicalChart = new Chart(ctx, {
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
                title: {
                    display: true,
                    text: 'Data source: Yahoo Finance API',
                    position: 'bottom',
                    align: 'start',
                    font: {
                        size: 9,
                        style: 'italic'
                    },
                    color: '#a0aec0',
                    padding: {
                        top: 10,
                        bottom: 0
                    }
                },
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 15,
                        font: {
                            size: 12,
                            weight: 'bold'
                        }
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
                                label += '$' + context.parsed.y.toLocaleString(undefined, {
                                    minimumFractionDigits: 2,
                                    maximumFractionDigits: 2
                                });
                            }
                            return label;
                        }
                    }
                }
            },
            scales: {
                ...yAxes,
                x: {
                    ticks: {
                        maxRotation: 45,
                        minRotation: 45,
                        maxTicksLimit: 7
                    },
                    title: {
                        display: true,
                        text: 'Date'
                    }
                }
            }
        }
    });
    console.log('Chart created successfully');
    } catch (error) {
        console.error('Error creating chart:', error);
    }
}

// Display predictions in right column
function displayPredictions(data) {
    const predictionsContent = document.getElementById('predictions-content');
    predictionsContent.innerHTML = '';
    
    if (!data.predictions || Object.keys(data.predictions).length === 0) {
        predictionsContent.innerHTML = '<div class="loading">No predictions available</div>';
        return;
    }
    
    // Create predictions grid
    const grid = document.createElement('div');
    grid.className = 'predictions-grid';
    
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
        
        grid.appendChild(card);
    }
    
    predictionsContent.appendChild(grid);
}

// Create predictions chart for all assets
function createPredictionsChart(data, period = 'weekly') {
    console.log('Creating predictions chart for period:', period);
    const canvas = document.getElementById('predictionsChart');
    if (!canvas) {
        console.error('Predictions canvas element not found');
        return;
    }
    
    const ctx = canvas.getContext('2d');
    if (!ctx) {
        console.error('Could not get predictions canvas context');
        return;
    }
    
    // Destroy existing chart if it exists
    if (predictionsChart) {
        predictionsChart.destroy();
    }
    
    // Get prediction data for the selected period
    const predictionsData = data.predictions_data?.[period];
    if (!predictionsData) {
        console.error('No predictions data available');
        return;
    }
    
    console.log('Predictions data:', predictionsData);
    
    // Colors for each asset
    const colors = {
        'Gold': '#f59e0b',
        'Bitcoin': '#f97316',
        'Oil': '#0ea5e9',
        'S&P 500': '#8b5cf6'
    };
    
    // Create datasets
    const datasets = [];
    const assetNames = Object.keys(predictionsData);
    
    // Get common date range
    let allDates = [];
    for (const assetName of assetNames) {
        const asset = predictionsData[assetName];
        if (asset.dates && asset.dates.length > 0) {
            allDates = asset.dates;
            break;
        }
    }
    
    for (const assetName of assetNames) {
        const asset = predictionsData[assetName];
        if (!asset.prices || asset.prices.length === 0) continue;
        
        datasets.push({
            label: assetName,
            data: asset.prices,
            borderColor: colors[assetName] || '#667eea',
            backgroundColor: 'transparent',
            borderWidth: 2.5,
            borderDash: [5, 5],
            pointRadius: 0,
            pointHoverRadius: 0,
            tension: 0.4,
            yAxisID: assetName.replace(/[^a-zA-Z0-9]/g, '')
        });
    }
    
    // Create y-axes configuration for each asset (same as historical chart)
    const yAxes = {};
    assetNames.forEach((assetName, index) => {
        const axisId = assetName.replace(/[^a-zA-Z0-9]/g, '');
        const isRightSide = index % 2 === 1;
        yAxes[axisId] = {
            type: 'linear',
            display: true,
            position: isRightSide ? 'right' : 'left',
            grid: {
                drawOnChartArea: index === 0,
            },
            title: {
                display: true,
                text: assetName,
                color: colors[assetName] || '#667eea'
            },
            ticks: {
                color: colors[assetName] || '#667eea',
                callback: function(value) {
                    return '$' + value.toLocaleString();
                }
            }
        };
    });
    
    if (typeof Chart === 'undefined') {
        console.error('Chart.js library not loaded');
        return;
    }
    
    try {
        predictionsChart = new Chart(ctx, {
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
                            padding: 15,
                            font: {
                                size: 12,
                                weight: 'bold'
                            }
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
                                    label += '$' + context.parsed.y.toLocaleString(undefined, {
                                        minimumFractionDigits: 2,
                                        maximumFractionDigits: 2
                                    });
                                }
                                return label;
                            }
                        }
                    }
                },
                scales: {
                    ...yAxes,
                    x: {
                        ticks: {
                            maxRotation: 45,
                            minRotation: 45,
                            maxTicksLimit: 7
                        },
                        title: {
                            display: true,
                            text: 'Date'
                        }
                    }
                }
            }
        });
        console.log('Predictions chart created successfully');
    } catch (error) {
        console.error('Error creating predictions chart:', error);
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
