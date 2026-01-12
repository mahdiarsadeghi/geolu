/**
 * Geolu - Where Algorithms Predict Value
 * Copyright (c) 2026 Geolu
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
        loadEvaluationResults();
    } catch (error) {
        console.error('Error loading data:', error);
    }
}

// Setup tab button listeners
function setupTabListeners() {
    const tabButtons = document.querySelectorAll('.tab-button');
    console.log('Setting up tab listeners, found', tabButtons.length, 'buttons');
    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            console.log('Button clicked:', this.getAttribute('data-period'));
            // Remove active class from all buttons
            tabButtons.forEach(btn => btn.classList.remove('active'));
            // Add active class to clicked button
            this.classList.add('active');
            // Update charts
            currentPeriod = this.getAttribute('data-period');
            createHistoricalChart(chartData, currentPeriod);
            createPredictionsChart(chartData, currentPeriod);
            // Highlight table column
            highlightTableColumn(currentPeriod);
        });
    });
    // Highlight initial column
    highlightTableColumn(currentPeriod);
}

// Highlight table column based on selected period
function highlightTableColumn(period) {
    // Remove all highlights
    document.querySelectorAll('.performance-table .highlight').forEach(el => {
        el.classList.remove('highlight');
    });
    
    // Add highlight to selected column
    const columnClass = 'acc-' + period;
    document.querySelectorAll('.' + columnClass).forEach(el => {
        el.classList.add('highlight');
    });
    
    // Highlight header
    const header = document.querySelector(`.time-period[data-period="${period}"]`);
    if (header) {
        header.classList.add('highlight');
    }
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
    
    // Use Bachata predictions if available, otherwise fall back to simple predictions
    let predictionsData;
    if (data.bachata_predictions && Object.keys(data.bachata_predictions).length > 0) {
        console.log('Using Bachata Fourier predictions');
        // Transform bachata_predictions structure: {Gold: {weekly: {...}, monthly: {...}}}
        predictionsData = {};
        for (const assetName in data.bachata_predictions) {
            const assetPredictions = data.bachata_predictions[assetName];
            if (assetPredictions[period]) {
                predictionsData[assetName] = assetPredictions[period];
            }
        }
    } else {
        console.log('Using simple baseline predictions');
        predictionsData = data.predictions_data?.[period];
    }
    
    if (!predictionsData || Object.keys(predictionsData).length === 0) {
        console.error('No predictions data available for period:', period);
        console.log('Available data:', data);
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
        console.error('Error details:', error.message, error.stack);
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

// Load evaluation results from CSV
async function loadEvaluationResults() {
    try {
        console.log('Loading evaluation results...');
        const response = await fetch('results.csv');
        if (!response.ok) {
            console.warn('Could not load evaluation results');
            return;
        }
        
        const csvText = await response.text();
        const lines = csvText.trim().split('\n');
        
        // Skip header
        const dataLines = lines.slice(1);
        
        // Parse CSV and calculate weighted averages per index per time scale
        const results = {
            'Gold': { 'weekly': [], 'monthly': [], 'yearly': [] },
            'Bitcoin': { 'weekly': [], 'monthly': [], 'yearly': [] },
            'Oil': { 'weekly': [], 'monthly': [], 'yearly': [] },
            'S&P 500': { 'weekly': [], 'monthly': [], 'yearly': [] }
        };
        
        dataLines.forEach(line => {
            const parts = line.split(',');
            if (parts.length >= 7) {
                const window = parts[1]; // Window (e.g., "4y-3y", "4y-2y", "4y-1y")
                const scale = parts[2]; // Prediction_Scale
                const goldError = parts[3]; // Gold_Error_%
                const bitcoinError = parts[4]; // Bitcoin_Error_%
                const oilError = parts[5]; // Oil_Error_%
                const stockError = parts[6]; // Stock_Error_%
                
                // Determine weight based on training data duration
                let weight = 1;
                if (window === '4y-3y') weight = 1; // 1 year of training data
                else if (window === '4y-2y') weight = 2; // 2 years of training data
                else if (window === '4y-1y') weight = 3; // 3 years of training data
                
                // Extract numeric value from percentage string (e.g., "1.91%" -> 1.91)
                const parseError = (str) => {
                    if (!str) return null;
                    const num = parseFloat(str.replace('%', ''));
                    return isNaN(num) ? null : num;
                };
                
                const goldVal = parseError(goldError);
                const bitcoinVal = parseError(bitcoinError);
                const oilVal = parseError(oilError);
                const stockVal = parseError(stockError);
                
                // Store value with weight as [value, weight]
                if (goldVal !== null) results['Gold'][scale].push([goldVal, weight]);
                if (bitcoinVal !== null) results['Bitcoin'][scale].push([bitcoinVal, weight]);
                if (oilVal !== null) results['Oil'][scale].push([oilVal, weight]);
                if (stockVal !== null) results['S&P 500'][scale].push([stockVal, weight]);
            }
        });
        
        // Calculate weighted averages and update table
        const indices = ['Gold', 'Bitcoin', 'Oil', 'S&P 500'];
        const scales = ['weekly', 'monthly', 'yearly'];
        
        indices.forEach(index => {
            scales.forEach(scale => {
                const data = results[index][scale];
                if (data.length > 0) {
                    // Calculate weighted average
                    let weightedSum = 0;
                    let totalWeight = 0;
                    
                    data.forEach(([value, weight]) => {
                        weightedSum += value * weight;
                        totalWeight += weight;
                    });
                    
                    const weightedAvg = weightedSum / totalWeight;
                    const cell = document.querySelector(`.acc-${scale}[data-index="${index}"]`);
                    if (cell) {
                        // Format to 3 significant figures
                        cell.textContent = weightedAvg.toPrecision(3) + '%';
                    }
                }
            });
        });
        
        console.log('Evaluation results loaded successfully');
    } catch (error) {
        console.error('Error loading evaluation results:', error);
    }
}

// Load data on page load
document.addEventListener('DOMContentLoaded', loadData);
