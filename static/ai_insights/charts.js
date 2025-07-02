// static/ai_insights/charts.js - Chart.js Configurations for AI Insights Dashboard

// ============================
// CHART.JS GLOBAL CONFIGURATION
// ============================

// Set Chart.js defaults for AI theme
Chart.defaults.font.family = "'Inter', -apple-system, BlinkMacSystemFont, sans-serif";
Chart.defaults.font.size = 12;
Chart.defaults.color = 'rgba(255, 255, 255, 0.8)';
Chart.defaults.backgroundColor = 'rgba(0, 212, 255, 0.1)';
Chart.defaults.borderColor = 'rgba(0, 212, 255, 0.3)';
Chart.defaults.scale.grid.color = 'rgba(255, 255, 255, 0.1)';
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.plugins.legend.labels.padding = 20;

// AI-themed color palette
const AI_COLORS = {
    primary: '#00d4ff',
    secondary: '#7c3aed', 
    success: '#00ff88',
    warning: '#ffc107',
    danger: '#ff6b6b',
    info: '#17a2b8',
    light: 'rgba(255, 255, 255, 0.8)',
    dark: 'rgba(15, 15, 25, 0.8)',
    
    // Gradient colors for charts
    gradients: {
        blue: ['#00d4ff', '#0099cc'],
        purple: ['#7c3aed', '#5b21b6'],
        green: ['#00ff88', '#00cc6a'],
        orange: ['#ff9f43', '#ee5a24'],
        red: ['#ff6b6b', '#e55353'],
        cyan: ['#00d4ff', '#00a8cc']
    },
    
    // Chart-specific color schemes
    priority: {
        high: '#00ff88',
        medium: '#ffc107', 
        low: '#ff6b6b',
        critical: '#e74c3c'
    },
    
    quality: {
        excellent: '#00ff88',
        good: '#00d4ff',
        fair: '#ffc107',
        poor: '#ff6b6b'
    },
    
    confidence: {
        high: '#00ff88',
        medium: '#00d4ff',
        low: '#ffc107'
    }
};

// ============================
// UTILITY FUNCTIONS
// ============================

/**
 * Create gradient background for charts
 */
function createGradient(ctx, color1, color2, vertical = true) {
    const gradient = ctx.createLinearGradient(0, 0, vertical ? 0 : ctx.canvas.width, vertical ? ctx.canvas.height : 0);
    gradient.addColorStop(0, color1);
    gradient.addColorStop(1, color2);
    return gradient;
}

/**
 * Create radial gradient for doughnut charts
 */
function createRadialGradient(ctx, color1, color2) {
    const centerX = ctx.canvas.width / 2;
    const centerY = ctx.canvas.height / 2;
    const radius = Math.min(centerX, centerY);
    
    const gradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, radius);
    gradient.addColorStop(0, color1);
    gradient.addColorStop(1, color2);
    return gradient;
}

/**
 * Format numbers for display
 */
function formatNumber(num) {
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
}

/**
 * Format percentage values
 */
function formatPercentage(value, decimals = 1) {
    return (value * 100).toFixed(decimals) + '%';
}

/**
 * Get responsive font size based on screen width
 */
function getResponsiveFontSize(baseSize = 12) {
    const screenWidth = window.innerWidth;
    if (screenWidth < 768) return Math.max(baseSize - 2, 10);
    if (screenWidth < 1024) return Math.max(baseSize - 1, 11);
    return baseSize;
}

// ============================
// AI PERFORMANCE CHARTS
// ============================

/**
 * AI Model Performance Radar Chart
 */
function createAIPerformanceRadar(ctx, data) {
    return new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Confidence', 'Speed'],
            datasets: [{
                label: 'Lead Priority Model',
                data: [
                    data.priority_model?.accuracy || 0,
                    data.priority_model?.precision || 0,
                    data.priority_model?.recall || 0,
                    data.priority_model?.f1_score || 0,
                    data.priority_model?.confidence || 0,
                    data.priority_model?.speed_score || 0
                ],
                backgroundColor: 'rgba(0, 212, 255, 0.2)',
                borderColor: AI_COLORS.primary,
                borderWidth: 2,
                pointBackgroundColor: AI_COLORS.primary,
                pointBorderColor: '#fff',
                pointBorderWidth: 2,
                pointRadius: 5
            }, {
                label: 'Quality Assessment Model',
                data: [
                    data.quality_model?.accuracy || 0,
                    data.quality_model?.precision || 0,
                    data.quality_model?.recall || 0,
                    data.quality_model?.f1_score || 0,
                    data.quality_model?.confidence || 0,
                    data.quality_model?.speed_score || 0
                ],
                backgroundColor: 'rgba(0, 255, 136, 0.2)',
                borderColor: AI_COLORS.success,
                borderWidth: 2,
                pointBackgroundColor: AI_COLORS.success,
                pointBorderColor: '#fff',
                pointBorderWidth: 2,
                pointRadius: 5
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'AI Model Performance Metrics',
                    font: { size: 16, weight: 'bold' },
                    color: AI_COLORS.light
                },
                legend: {
                    position: 'bottom',
                    labels: { color: AI_COLORS.light }
                }
            },
            scales: {
                r: {
                    beginAtZero: true,
                    max: 1.0,
                    ticks: {
                        stepSize: 0.2,
                        color: AI_COLORS.light,
                        backdropColor: 'transparent'
                    },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    pointLabels: { 
                        color: AI_COLORS.light,
                        font: { size: getResponsiveFontSize(11) }
                    }
                }
            }
        }
    });
}

/**
 * Lead Priority Distribution Doughnut Chart
 */
function createPriorityDistributionChart(ctx, data) {
    const priorityData = data.priority_distribution || {};
    const labels = Object.keys(priorityData);
    const values = Object.values(priorityData);
    
    return new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels.map(label => label.charAt(0).toUpperCase() + label.slice(1)),
            datasets: [{
                data: values,
                backgroundColor: [
                    AI_COLORS.priority.critical,
                    AI_COLORS.priority.high,
                    AI_COLORS.priority.medium,
                    AI_COLORS.priority.low
                ],
                borderColor: 'rgba(255, 255, 255, 0.2)',
                borderWidth: 2,
                hoverBorderWidth: 3,
                hoverBorderColor: '#fff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '60%',
            plugins: {
                title: {
                    display: true,
                    text: 'Lead Priority Distribution',
                    font: { size: 16, weight: 'bold' },
                    color: AI_COLORS.light
                },
                legend: {
                    position: 'bottom',
                    labels: { 
                        color: AI_COLORS.light,
                        usePointStyle: true,
                        padding: 15
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = ((context.parsed / total) * 100).toFixed(1);
                            return `${context.label}: ${context.parsed} (${percentage}%)`;
                        }
                    }
                }
            }
        }
    });
}

/**
 * Data Quality Trends Line Chart
 */
function createQualityTrendsChart(ctx, data) {
    const trendData = data.quality_trends || [];
    
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: trendData.map(item => new Date(item.date).toLocaleDateString()),
            datasets: [{
                label: 'Average Quality Score',
                data: trendData.map(item => item.quality_score),
                borderColor: AI_COLORS.success,
                backgroundColor: function(context) {
                    const ctx = context.chart.ctx;
                    return createGradient(ctx, 'rgba(0, 255, 136, 0.3)', 'rgba(0, 255, 136, 0.05)');
                },
                borderWidth: 3,
                fill: true,
                tension: 0.4,
                pointBackgroundColor: AI_COLORS.success,
                pointBorderColor: '#fff',
                pointBorderWidth: 2,
                pointRadius: 5,
                pointHoverRadius: 7
            }, {
                label: 'Completeness Score',
                data: trendData.map(item => item.completeness_score || 0),
                borderColor: AI_COLORS.primary,
                backgroundColor: 'rgba(0, 212, 255, 0.1)',
                borderWidth: 2,
                fill: false,
                tension: 0.4,
                pointBackgroundColor: AI_COLORS.primary,
                pointBorderColor: '#fff',
                pointBorderWidth: 2,
                pointRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Data Quality Trends (30 Days)',
                    font: { size: 16, weight: 'bold' },
                    color: AI_COLORS.light
                },
                legend: {
                    position: 'top',
                    labels: { color: AI_COLORS.light }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { color: AI_COLORS.light }
                },
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { 
                        color: AI_COLORS.light,
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            }
        }
    });
}

/**
 * Prediction Confidence Distribution Bar Chart
 */
function createConfidenceDistributionChart(ctx, data) {
    const confidenceData = data.confidence_distribution || {};
    
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['High (90-100%)', 'Medium (70-90%)', 'Low (50-70%)', 'Very Low (<50%)'],
            datasets: [{
                label: 'Number of Predictions',
                data: [
                    confidenceData.high || 0,
                    confidenceData.medium || 0,
                    confidenceData.low || 0,
                    confidenceData.very_low || 0
                ],
                backgroundColor: [
                    AI_COLORS.confidence.high,
                    AI_COLORS.confidence.medium,
                    AI_COLORS.confidence.low,
                    AI_COLORS.danger
                ],
                borderColor: 'rgba(255, 255, 255, 0.2)',
                borderWidth: 1,
                borderRadius: 8,
                borderSkipped: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'AI Prediction Confidence Distribution',
                    font: { size: 16, weight: 'bold' },
                    color: AI_COLORS.light
                },
                legend: { display: false }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { 
                        color: AI_COLORS.light,
                        font: { size: getResponsiveFontSize(10) }
                    }
                },
                y: {
                    beginAtZero: true,
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { 
                        color: AI_COLORS.light,
                        callback: function(value) {
                            return formatNumber(value);
                        }
                    }
                }
            }
        }
    });
}

/**
 * Feature Importance Horizontal Bar Chart
 */
function createFeatureImportanceChart(ctx, data) {
    const featureData = data.feature_importance || {};
    const features = Object.keys(featureData).slice(0, 10); // Top 10 features
    const importance = features.map(feature => featureData[feature]);
    
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: features.map(feature => feature.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())),
            datasets: [{
                label: 'Importance Score',
                data: importance,
                backgroundColor: function(context) {
                    const value = context.parsed.x;
                    const maxValue = Math.max(...importance);
                    const intensity = value / maxValue;
                    return `rgba(0, 212, 255, ${0.3 + intensity * 0.7})`;
                },
                borderColor: AI_COLORS.primary,
                borderWidth: 2,
                borderRadius: 6,
                borderSkipped: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            plugins: {
                title: {
                    display: true,
                    text: 'Top Feature Importance (Lead Priority Model)',
                    font: { size: 16, weight: 'bold' },
                    color: AI_COLORS.light
                },
                legend: { display: false }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { 
                        color: AI_COLORS.light,
                        callback: function(value) {
                            return formatPercentage(value, 1);
                        }
                    }
                },
                y: {
                    grid: { display: false },
                    ticks: { 
                        color: AI_COLORS.light,
                        font: { size: getResponsiveFontSize(10) }
                    }
                }
            }
        }
    });
}

// ============================
// MODEL PERFORMANCE CHARTS
// ============================

/**
 * Model Training Progress Chart
 */
function createTrainingProgressChart(ctx, data) {
    const trainingData = data.training_history || [];
    
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: trainingData.map((_, index) => `Epoch ${index + 1}`),
            datasets: [{
                label: 'Training Accuracy',
                data: trainingData.map(item => item.train_accuracy),
                borderColor: AI_COLORS.primary,
                backgroundColor: 'rgba(0, 212, 255, 0.1)',
                borderWidth: 2,
                tension: 0.4,
                pointRadius: 3
            }, {
                label: 'Validation Accuracy',
                data: trainingData.map(item => item.val_accuracy),
                borderColor: AI_COLORS.success,
                backgroundColor: 'rgba(0, 255, 136, 0.1)',
                borderWidth: 2,
                tension: 0.4,
                pointRadius: 3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Model Training Progress',
                    font: { size: 16, weight: 'bold' },
                    color: AI_COLORS.light
                },
                legend: {
                    position: 'top',
                    labels: { color: AI_COLORS.light }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { color: AI_COLORS.light }
                },
                y: {
                    beginAtZero: true,
                    max: 1.0,
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { 
                        color: AI_COLORS.light,
                        callback: function(value) {
                            return formatPercentage(value);
                        }
                    }
                }
            }
        }
    });
}

/**
 * Prediction Accuracy Over Time
 */
function createAccuracyOverTimeChart(ctx, data) {
    const accuracyData = data.accuracy_timeline || [];
    
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: accuracyData.map(item => new Date(item.date).toLocaleDateString()),
            datasets: [{
                label: 'Priority Predictions',
                data: accuracyData.map(item => item.priority_accuracy),
                borderColor: AI_COLORS.primary,
                backgroundColor: function(context) {
                    const ctx = context.chart.ctx;
                    return createGradient(ctx, 'rgba(0, 212, 255, 0.3)', 'rgba(0, 212, 255, 0.05)');
                },
                fill: true,
                tension: 0.4,
                borderWidth: 3,
                pointRadius: 4
            }, {
                label: 'Quality Predictions',
                data: accuracyData.map(item => item.quality_accuracy),
                borderColor: AI_COLORS.success,
                backgroundColor: function(context) {
                    const ctx = context.chart.ctx;
                    return createGradient(ctx, 'rgba(0, 255, 136, 0.3)', 'rgba(0, 255, 136, 0.05)');
                },
                fill: true,
                tension: 0.4,
                borderWidth: 3,
                pointRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Prediction Accuracy Over Time',
                    font: { size: 16, weight: 'bold' },
                    color: AI_COLORS.light
                },
                legend: {
                    position: 'top',
                    labels: { color: AI_COLORS.light }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { color: AI_COLORS.light }
                },
                y: {
                    beginAtZero: true,
                    max: 1.0,
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { 
                        color: AI_COLORS.light,
                        callback: function(value) {
                            return formatPercentage(value);
                        }
                    }
                }
            }
        }
    });
}

// ============================
// BUSINESS IMPACT CHARTS
// ============================

/**
 * Conversion Rate by AI Priority
 */
function createConversionByPriorityChart(ctx, data) {
    const conversionData = data.conversion_by_priority || {};
    
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['High Priority', 'Medium Priority', 'Low Priority'],
            datasets: [{
                label: 'Conversion Rate',
                data: [
                    conversionData.high || 0,
                    conversionData.medium || 0,
                    conversionData.low || 0
                ],
                backgroundColor: [
                    AI_COLORS.priority.high,
                    AI_COLORS.priority.medium,
                    AI_COLORS.priority.low
                ],
                borderColor: 'rgba(255, 255, 255, 0.3)',
                borderWidth: 2,
                borderRadius: 8,
                borderSkipped: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Conversion Rate by AI Priority',
                    font: { size: 16, weight: 'bold' },
                    color: AI_COLORS.light
                },
                legend: { display: false }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { color: AI_COLORS.light }
                },
                y: {
                    beginAtZero: true,
                    max: 1.0,
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { 
                        color: AI_COLORS.light,
                        callback: function(value) {
                            return formatPercentage(value);
                        }
                    }
                }
            }
        }
    });
}

/**
 * AI Insights Impact Metrics
 */
function createImpactMetricsChart(ctx, data) {
    const metrics = data.impact_metrics || {};
    
    return new Chart(ctx, {
        type: 'polarArea',
        data: {
            labels: ['Time Saved', 'Quality Improved', 'Leads Prioritized', 'Accuracy Gained', 'ROI Increase'],
            datasets: [{
                data: [
                    metrics.time_saved_percentage || 0,
                    metrics.quality_improvement || 0,
                    metrics.leads_prioritized_percentage || 0,
                    metrics.accuracy_improvement || 0,
                    metrics.roi_increase || 0
                ],
                backgroundColor: [
                    'rgba(0, 212, 255, 0.6)',
                    'rgba(0, 255, 136, 0.6)',
                    'rgba(124, 58, 237, 0.6)',
                    'rgba(255, 193, 7, 0.6)',
                    'rgba(255, 107, 107, 0.6)'
                ],
                borderColor: [
                    AI_COLORS.primary,
                    AI_COLORS.success,
                    AI_COLORS.secondary,
                    AI_COLORS.warning,
                    AI_COLORS.danger
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'AI Business Impact Metrics',
                    font: { size: 16, weight: 'bold' },
                    color: AI_COLORS.light
                },
                legend: {
                    position: 'bottom',
                    labels: { 
                        color: AI_COLORS.light,
                        usePointStyle: true
                    }
                }
            },
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        color: AI_COLORS.light,
                        backdropColor: 'transparent',
                        callback: function(value) {
                            return value + '%';
                        }
                    },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' }
                }
            }
        }
    });
}

// ============================
// REAL-TIME MONITORING CHARTS
// ============================

/**
 * Real-time AI Processing Queue
 */
function createProcessingQueueChart(ctx, data) {
    const queueData = data.processing_queue || [];
    
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: queueData.map(item => new Date(item.timestamp).toLocaleTimeString()),
            datasets: [{
                label: 'Queue Length',
                data: queueData.map(item => item.queue_length),
                borderColor: AI_COLORS.warning,
                backgroundColor: 'rgba(255, 193, 7, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.1,
                pointRadius: 2
            }, {
                label: 'Processing Rate',
                data: queueData.map(item => item.processing_rate),
                borderColor: AI_COLORS.success,
                backgroundColor: 'rgba(0, 255, 136, 0.1)',
                borderWidth: 2,
                fill: false,
                tension: 0.1,
                pointRadius: 2,
                yAxisID: 'y1'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Real-time AI Processing Queue',
                    font: { size: 16, weight: 'bold' },
                    color: AI_COLORS.light
                },
                legend: {
                    position: 'top',
                    labels: { color: AI_COLORS.light }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { color: AI_COLORS.light }
                },
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    beginAtZero: true,
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { color: AI_COLORS.light }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    beginAtZero: true,
                    grid: { drawOnChartArea: false },
                    ticks: { color: AI_COLORS.light }
                }
            }
        }
    });
}

// ============================
// CHART MANAGEMENT FUNCTIONS
// ============================

/**
 * Initialize all AI insight charts
 */
function initializeAICharts(containerId, data) {
    const container = document.getElementById(containerId);
    if (!container) {
        console.error('Charts container not found:', containerId);
        return;
    }

    // Create chart containers
    const chartConfigs = [
        { id: 'ai-performance-radar', title: 'AI Model Performance', type: 'radar' },
        { id: 'priority-distribution', title: 'Priority Distribution', type: 'doughnut' },
        { id: 'quality-trends', title: 'Quality Trends', type: 'line' },
        { id: 'confidence-distribution', title: 'Confidence Distribution', type: 'bar' },
        { id: 'feature-importance', title: 'Feature Importance', type: 'bar' },
        { id: 'conversion-by-priority', title: 'Conversion by Priority', type: 'bar' },
        { id: 'impact-metrics', title: 'Impact Metrics', type: 'polarArea' },
        { id: 'accuracy-timeline', title: 'Accuracy Timeline', type: 'line' }
    ];

    chartConfigs.forEach(config => {
        const chartContainer = document.createElement('div');
        chartContainer.className = 'chart-container';
        chartContainer.innerHTML = `
            <div class="chart-header">
                <h3>${config.title}</h3>
                <div class="chart-controls">
                    <button class="chart-refresh-btn" onclick="refreshChart('${config.id}')">
                        <i class="fas fa-sync-alt"></i>
                    </button>
                    <button class="chart-fullscreen-btn" onclick="toggleChartFullscreen('${config.id}')">
                        <i class="fas fa-expand"></i>
                    </button>
                </div>
            <div class="chart-wrapper">
                <canvas id="${config.id}" width="400" height="300"></canvas>
            </div>
        `;
        container.appendChild(chartContainer);
    });

    // Initialize individual charts
    const charts = {};
    
    try {
        charts.performanceRadar = createAIPerformanceRadar(
            document.getElementById('ai-performance-radar').getContext('2d'), 
            data
        );
        
        charts.priorityDistribution = createPriorityDistributionChart(
            document.getElementById('priority-distribution').getContext('2d'), 
            data
        );
        
        charts.qualityTrends = createQualityTrendsChart(
            document.getElementById('quality-trends').getContext('2d'), 
            data
        );
        
        charts.confidenceDistribution = createConfidenceDistributionChart(
            document.getElementById('confidence-distribution').getContext('2d'), 
            data
        );
        
        charts.featureImportance = createFeatureImportanceChart(
            document.getElementById('feature-importance').getContext('2d'), 
            data
        );
        
        charts.conversionByPriority = createConversionByPriorityChart(
            document.getElementById('conversion-by-priority').getContext('2d'), 
            data
        );
        
        charts.impactMetrics = createImpactMetricsChart(
            document.getElementById('impact-metrics').getContext('2d'), 
            data
        );
        
        charts.accuracyTimeline = createAccuracyOverTimeChart(
            document.getElementById('accuracy-timeline').getContext('2d'), 
            data
        );

    } catch (error) {
        console.error('Error initializing charts:', error);
    }

    return charts;
}

/**
 * Update chart data dynamically
 */
function updateChart(chartInstance, newData) {
    if (!chartInstance) return;
    
    try {
        chartInstance.data = newData;
        chartInstance.update('none'); // No animation for real-time updates
    } catch (error) {
        console.error('Error updating chart:', error);
    }
}

/**
 * Refresh individual chart
 */
function refreshChart(chartId) {
    const chartElement = document.getElementById(chartId);
    if (!chartElement) return;
    
    // Add refresh animation
    chartElement.style.opacity = '0.5';
    
    // Simulate data refresh (replace with actual API call)
    setTimeout(() => {
        chartElement.style.opacity = '1';
        // In real implementation, fetch new data and update chart
        console.log(`Refreshing chart: ${chartId}`);
    }, 500);
}

/**
 * Toggle chart fullscreen mode
 */
function toggleChartFullscreen(chartId) {
    const chartContainer = document.getElementById(chartId).closest('.chart-container');
    if (!chartContainer) return;
    
    chartContainer.classList.toggle('fullscreen');
    
    // Update chart size after transition
    setTimeout(() => {
        const chart = Chart.getChart(chartId);
        if (chart) {
            chart.resize();
        }
    }, 300);
}

/**
 * Export chart as image
 */
function exportChart(chartId, filename) {
    const chart = Chart.getChart(chartId);
    if (!chart) return;
    
    const url = chart.toBase64Image();
    const link = document.createElement('a');
    link.download = filename || `${chartId}.png`;
    link.href = url;
    link.click();
}

// ============================
// RESPONSIVE CHART HANDLING
// ============================

/**
 * Handle window resize for all charts
 */
function handleChartResize() {
    Chart.helpers.each(Chart.instances, function(instance) {
        instance.resize();
    });
}

// Add resize listener
window.addEventListener('resize', debounce(handleChartResize, 250));

/**
 * Debounce function for performance
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// ============================
// ANIMATION AND INTERACTIONS
// ============================

/**
 * Custom animation configurations
 */
const CHART_ANIMATIONS = {
    fadeIn: {
        animation: {
            duration: 1000,
            easing: 'easeInOutQuart'
        }
    },
    
    slideUp: {
        animation: {
            duration: 800,
            easing: 'easeOutBounce',
            onProgress: function(animation) {
                const progress = animation.currentStep / animation.numSteps;
                this.chart.canvas.style.transform = `translateY(${50 * (1 - progress)}px)`;
                this.chart.canvas.style.opacity = progress;
            },
            onComplete: function() {
                this.chart.canvas.style.transform = 'translateY(0)';
                this.chart.canvas.style.opacity = '1';
            }
        }
    },
    
    pulse: {
        animation: {
            duration: 1200,
            easing: 'easeInOutSine',
            loop: false
        }
    }
};

/**
 * Add hover effects to chart elements
 */
function addChartHoverEffects() {
    const chartContainers = document.querySelectorAll('.chart-container');
    
    chartContainers.forEach(container => {
        const canvas = container.querySelector('canvas');
        
        container.addEventListener('mouseenter', () => {
            container.style.transform = 'scale(1.02)';
            container.style.boxShadow = '0 10px 30px rgba(0, 212, 255, 0.3)';
        });
        
        container.addEventListener('mouseleave', () => {
            container.style.transform = 'scale(1)';
            container.style.boxShadow = '0 5px 15px rgba(0, 0, 0, 0.3)';
        });
    });
}

// ============================
// DATA PROCESSING UTILITIES
// ============================

/**
 * Process API data for charts
 */
function processAIInsightData(rawData) {
    return {
        priority_distribution: {
            high: rawData.high_priority_count || 0,
            medium: rawData.medium_priority_count || 0,
            low: rawData.low_priority_count || 0,
            critical: rawData.critical_priority_count || 0
        },
        
        quality_trends: (rawData.data_quality_trend || []).map(item => ({
            date: item.date,
            quality_score: item.quality || 0,
            completeness_score: item.completeness || 0,
            accuracy_score: item.accuracy || 0
        })),
        
        confidence_distribution: {
            high: rawData.high_confidence_predictions || 0,
            medium: rawData.medium_confidence_predictions || 0,
            low: rawData.low_confidence_predictions || 0,
            very_low: rawData.very_low_confidence_predictions || 0
        },
        
        feature_importance: rawData.feature_importance || {},
        
        conversion_by_priority: {
            high: rawData.high_priority_conversion_rate || 0,
            medium: rawData.medium_priority_conversion_rate || 0,
            low: rawData.low_priority_conversion_rate || 0
        },
        
        impact_metrics: {
            time_saved_percentage: rawData.time_saved_percentage || 0,
            quality_improvement: rawData.quality_improvement_percentage || 0,
            leads_prioritized_percentage: rawData.leads_prioritized_percentage || 0,
            accuracy_improvement: rawData.accuracy_improvement_percentage || 0,
            roi_increase: rawData.roi_increase_percentage || 0
        },
        
        accuracy_timeline: (rawData.accuracy_timeline || []).map(item => ({
            date: item.date,
            priority_accuracy: item.priority_accuracy || 0,
            quality_accuracy: item.quality_accuracy || 0
        })),
        
        priority_model: {
            accuracy: rawData.priority_model_accuracy || 0,
            precision: rawData.priority_model_precision || 0,
            recall: rawData.priority_model_recall || 0,
            f1_score: rawData.priority_model_f1 || 0,
            confidence: rawData.priority_model_confidence || 0,
            speed_score: rawData.priority_model_speed || 0
        },
        
        quality_model: {
            accuracy: rawData.quality_model_accuracy || 0,
            precision: rawData.quality_model_precision || 0,
            recall: rawData.quality_model_recall || 0,
            f1_score: rawData.quality_model_f1 || 0,
            confidence: rawData.quality_model_confidence || 0,
            speed_score: rawData.quality_model_speed || 0
        }
    };
}

/**
 * Generate sample data for testing
 */
function generateSampleAIData() {
    const now = new Date();
    const thirtyDaysAgo = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
    
    // Generate quality trends
    const qualityTrends = [];
    for (let i = 0; i < 30; i++) {
        const date = new Date(thirtyDaysAgo.getTime() + i * 24 * 60 * 60 * 1000);
        qualityTrends.push({
            date: date.toISOString(),
            quality_score: 70 + Math.random() * 25,
            completeness_score: 65 + Math.random() * 30,
            accuracy_score: 75 + Math.random() * 20
        });
    }
    
    // Generate accuracy timeline
    const accuracyTimeline = [];
    for (let i = 0; i < 14; i++) {
        const date = new Date(now.getTime() - (13 - i) * 24 * 60 * 60 * 1000);
        accuracyTimeline.push({
            date: date.toISOString(),
            priority_accuracy: 0.8 + Math.random() * 0.15,
            quality_accuracy: 0.75 + Math.random() * 0.2
        });
    }
    
    return {
        high_priority_count: 45,
        medium_priority_count: 123,
        low_priority_count: 67,
        critical_priority_count: 12,
        
        data_quality_trend: qualityTrends,
        
        high_confidence_predictions: 234,
        medium_confidence_predictions: 167,
        low_confidence_predictions: 89,
        very_low_confidence_predictions: 23,
        
        feature_importance: {
            email_quality: 0.15,
            title_seniority: 0.12,
            company_size: 0.11,
            industry_relevance: 0.10,
            source_quality: 0.09,
            data_completeness: 0.08,
            linkedin_presence: 0.07,
            phone_available: 0.06,
            recent_activity: 0.05,
            location_score: 0.04
        },
        
        high_priority_conversion_rate: 0.35,
        medium_priority_conversion_rate: 0.18,
        low_priority_conversion_rate: 0.08,
        
        time_saved_percentage: 65,
        quality_improvement_percentage: 40,
        leads_prioritized_percentage: 85,
        accuracy_improvement_percentage: 45,
        roi_increase_percentage: 120,
        
        accuracy_timeline: accuracyTimeline,
        
        priority_model_accuracy: 0.87,
        priority_model_precision: 0.84,
        priority_model_recall: 0.89,
        priority_model_f1: 0.86,
        priority_model_confidence: 0.91,
        priority_model_speed: 0.95,
        
        quality_model_accuracy: 0.82,
        quality_model_precision: 0.80,
        quality_model_recall: 0.85,
        quality_model_f1: 0.82,
        quality_model_confidence: 0.88,
        quality_model_speed: 0.92
    };
}

// ============================
// REAL-TIME UPDATES
// ============================

/**
 * Real-time chart update manager
 */
class RealTimeChartManager {
    constructor(updateInterval = 30000) { // 30 seconds
        this.updateInterval = updateInterval;
        this.isRunning = false;
        this.intervalId = null;
        this.charts = {};
    }
    
    start(charts) {
        if (this.isRunning) return;
        
        this.charts = charts;
        this.isRunning = true;
        
        this.intervalId = setInterval(() => {
            this.updateAllCharts();
        }, this.updateInterval);
        
        console.log('Real-time chart updates started');
    }
    
    stop() {
        if (!this.isRunning) return;
        
        clearInterval(this.intervalId);
        this.isRunning = false;
        
        console.log('Real-time chart updates stopped');
    }
    
    async updateAllCharts() {
        try {
            // Fetch latest data (replace with actual API call)
            const latestData = await this.fetchLatestData();
            const processedData = processAIInsightData(latestData);
            
            // Update each chart
            Object.keys(this.charts).forEach(chartKey => {
                this.updateSpecificChart(chartKey, processedData);
            });
            
        } catch (error) {
            console.error('Error updating charts:', error);
        }
    }
    
    async fetchLatestData() {
        // Simulate API call - replace with actual implementation
        return new Promise(resolve => {
            setTimeout(() => {
                resolve(generateSampleAIData());
            }, 100);
        });
    }
    
    updateSpecificChart(chartKey, data) {
        const chart = this.charts[chartKey];
        if (!chart) return;
        
        try {
            switch (chartKey) {
                case 'priorityDistribution':
                    chart.data.datasets[0].data = [
                        data.priority_distribution.critical,
                        data.priority_distribution.high,
                        data.priority_distribution.medium,
                        data.priority_distribution.low
                    ];
                    break;
                    
                case 'qualityTrends':
                    chart.data.labels = data.quality_trends.map(item => 
                        new Date(item.date).toLocaleDateString()
                    );
                    chart.data.datasets[0].data = data.quality_trends.map(item => item.quality_score);
                    chart.data.datasets[1].data = data.quality_trends.map(item => item.completeness_score);
                    break;
                    
                case 'confidenceDistribution':
                    chart.data.datasets[0].data = [
                        data.confidence_distribution.high,
                        data.confidence_distribution.medium,
                        data.confidence_distribution.low,
                        data.confidence_distribution.very_low
                    ];
                    break;
                    
                // Add more chart updates as needed
            }
            
            chart.update('none'); // Update without animation for real-time
            
        } catch (error) {
            console.error(`Error updating ${chartKey}:`, error);
        }
    }
}

// ============================
// CHART THEMES AND CUSTOMIZATION
// ============================

/**
 * Theme manager for charts
 */
class ChartThemeManager {
    constructor() {
        this.themes = {
            dark: {
                backgroundColor: 'rgba(15, 15, 25, 0.8)',
                textColor: 'rgba(255, 255, 255, 0.8)',
                gridColor: 'rgba(255, 255, 255, 0.1)',
                primaryColor: '#00d4ff',
                secondaryColor: '#7c3aed',
                successColor: '#00ff88',
                warningColor: '#ffc107',
                dangerColor: '#ff6b6b'
            },
            light: {
                backgroundColor: 'rgba(255, 255, 255, 0.9)',
                textColor: 'rgba(0, 0, 0, 0.8)',
                gridColor: 'rgba(0, 0, 0, 0.1)',
                primaryColor: '#007acc',
                secondaryColor: '#6366f1',
                successColor: '#10b981',
                warningColor: '#f59e0b',
                dangerColor: '#ef4444'
            }
        };
        
        this.currentTheme = 'dark';
    }
    
    applyTheme(themeName) {
        const theme = this.themes[themeName];
        if (!theme) return;
        
        this.currentTheme = themeName;
        
        // Update Chart.js defaults
        Chart.defaults.color = theme.textColor;
        Chart.defaults.backgroundColor = theme.backgroundColor;
        Chart.defaults.borderColor = theme.primaryColor;
        Chart.defaults.scale.grid.color = theme.gridColor;
        
        // Update AI_COLORS object
        Object.assign(AI_COLORS, {
            primary: theme.primaryColor,
            secondary: theme.secondaryColor,
            success: theme.successColor,
            warning: theme.warningColor,
            danger: theme.dangerColor,
            light: theme.textColor,
            dark: theme.backgroundColor
        });
        
        // Refresh all charts
        Chart.helpers.each(Chart.instances, function(instance) {
            instance.update();
        });
    }
}

// ============================
// PERFORMANCE OPTIMIZATION
// ============================

/**
 * Chart performance optimizer
 */
class ChartPerformanceOptimizer {
    constructor() {
        this.observedCharts = new Map();
        this.intersectionObserver = null;
        this.initIntersectionObserver();
    }
    
    initIntersectionObserver() {
        this.intersectionObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                const chartId = entry.target.id;
                const chart = Chart.getChart(chartId);
                
                if (chart) {
                    if (entry.isIntersecting) {
                        // Chart is visible, enable animations
                        chart.options.animation = { duration: 1000 };
                        this.observedCharts.set(chartId, true);
                    } else {
                        // Chart is not visible, disable animations for performance
                        chart.options.animation = { duration: 0 };
                        this.observedCharts.set(chartId, false);
                    }
                }
            });
        }, {
            rootMargin: '50px',
            threshold: 0.1
        });
    }
    
    observeChart(chartElement) {
        if (chartElement && this.intersectionObserver) {
            this.intersectionObserver.observe(chartElement);
        }
    }
    
    unobserveChart(chartElement) {
        if (chartElement && this.intersectionObserver) {
            this.intersectionObserver.unobserve(chartElement);
        }
    }
    
    optimizeChartData(data, maxDataPoints = 100) {
        if (!Array.isArray(data) || data.length <= maxDataPoints) {
            return data;
        }
        
        // Sample data to reduce points while maintaining trends
        const step = Math.ceil(data.length / maxDataPoints);
        return data.filter((_, index) => index % step === 0);
    }
}

// ============================
// INITIALIZATION
// ============================

// Global instances
let realTimeManager = null;
let themeManager = null;
let performanceOptimizer = null;

/**
 * Initialize AI Charts system
 */
function initializeAIChartsSystem(containerId, data) {
    // Initialize theme manager
    themeManager = new ChartThemeManager();
    
    // Initialize performance optimizer
    performanceOptimizer = new ChartPerformanceOptimizer();
    
    // Process data
    const processedData = data ? processAIInsightData(data) : processAIInsightData(generateSampleAIData());
    
    // Initialize charts
    const charts = initializeAICharts(containerId, processedData);
    
    // Add hover effects
    addChartHoverEffects();
    
    // Observe charts for performance
    Object.keys(charts).forEach(chartKey => {
        const chartElement = document.getElementById(chartKey.replace(/([A-Z])/g, '-$1').toLowerCase().replace(/^-/, ''));
        if (chartElement) {
            performanceOptimizer.observeChart(chartElement);
        }
    });
    
    // Initialize real-time updates
    realTimeManager = new RealTimeChartManager();
    realTimeManager.start(charts);
    
    console.log('AI Charts system initialized successfully');
    
    return {
        charts,
        realTimeManager,
        themeManager,
        performanceOptimizer
    };
}

/**
 * Cleanup AI Charts system
 */
function cleanupAIChartsSystem() {
    if (realTimeManager) {
        realTimeManager.stop();
    }
    
    // Destroy all charts
    Chart.helpers.each(Chart.instances, function(instance) {
        instance.destroy();
    });
    
    console.log('AI Charts system cleaned up');
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        initializeAIChartsSystem,
        cleanupAIChartsSystem,
        AI_COLORS,
        createAIPerformanceRadar,
        createPriorityDistributionChart,
        createQualityTrendsChart,
        RealTimeChartManager,
        ChartThemeManager,
        ChartPerformanceOptimizer
    };
}

// Auto-initialize if DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    const chartsContainer = document.getElementById('ai-charts-container');
    if (chartsContainer) {
        // Wait for potential data loading
        setTimeout(() => {
            initializeAIChartsSystem('ai-charts-container');
        }, 500);
    }
});

// Cleanup on page unload
window.addEventListener('beforeunload', function() {
    cleanupAIChartsSystem();
});