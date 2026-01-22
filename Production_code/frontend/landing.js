// Fetch real stats from backend
async function loadStats() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        
        const nodesElement = document.getElementById('stat-nodes');
        if (nodesElement && data.graph && data.graph.nodes) {
            const nodeCount = data.graph.nodes;
            // Animate counter if we have real data
            if (nodeCount > 100) {
                animateCounter(nodesElement, nodeCount);
            }
        }
    } catch (error) {
        console.error('Error loading stats:', error);
        // Keep default values if API fails
    }
}

function animateCounter(element, target) {
    const currentValue = element.textContent;
    const currentNum = parseFloat(currentValue.replace('K+', '')) * 1000 || 0;
    
    // Only animate if the new value is significantly different
    if (Math.abs(target - currentNum) < 100) {
        return;
    }
    
    const duration = 1500;
    const start = currentNum;
    const increment = (target - start) / (duration / 16);
    let current = start;
    
    const timer = setInterval(() => {
        current += increment;
        if ((increment > 0 && current >= target) || (increment < 0 && current <= target)) {
            current = target;
            clearInterval(timer);
        }
        element.textContent = formatNumber(Math.floor(current));
    }, 16);
}

function formatNumber(num) {
    if (num >= 1000) {
        return (num / 1000).toFixed(1).replace(/\.0$/, '') + 'K+';
    }
    return num.toLocaleString();
}

// Smooth scroll for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const targetId = this.getAttribute('href');
        if (targetId === '#') return;
        
        const target = document.querySelector(targetId);
        if (target) {
            window.scrollTo({
                top: target.offsetTop - 80,
                behavior: 'smooth'
            });
        }
    });
});

// Animate hero visual typing
function animateTyping() {
    const typingDots = document.querySelectorAll('.typing-animation span');
    if (typingDots.length > 0) {
        typingDots.forEach((dot, index) => {
            dot.style.animation = `typing 1.4s infinite ${index * 0.2}s`;
        });
    }
}

// Load stats when page loads
document.addEventListener('DOMContentLoaded', () => {
    loadStats();
    animateTyping();
    
    // Add subtle animation to stats cards
    const statCards = document.querySelectorAll('.stat-card');
    statCards.forEach((card, index) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        
        setTimeout(() => {
            card.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, 300 + (index * 100));
    });
});