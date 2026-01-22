// Tab switching
document.querySelectorAll('.auth-tab').forEach(tab => {
    tab.addEventListener('click', () => {
        const targetTab = tab.dataset.tab;
        
        document.querySelectorAll('.auth-tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        
        document.querySelectorAll('.auth-form').forEach(form => {
            form.style.display = 'none';
        });
        
        if (targetTab === 'login') {
            document.getElementById('login-form').style.display = 'flex';
        } else {
            document.getElementById('signup-form').style.display = 'flex';
        }
        
        document.getElementById('login-error').textContent = '';
        document.getElementById('signup-error').textContent = '';
    });
});

// Login form
document.getElementById('login-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const username = document.getElementById('login-username').value;
    const password = document.getElementById('login-password').value;
    const errorDiv = document.getElementById('login-error');
    
    try {
        const response = await fetch('/api/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password })
        });
        
        if (response.ok) {
            const data = await response.json();
            localStorage.setItem('user_id', data.user_id);
            localStorage.setItem('username', data.username);
            window.location.href = '/static/chat.html';
        } else {
            const error = await response.json();
            errorDiv.textContent = error.detail || 'Login failed';
        }
    } catch (error) {
        errorDiv.textContent = 'Connection error';
    }
});

// Signup form
document.getElementById('signup-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const fullname = document.getElementById('signup-fullname').value;
    const username = document.getElementById('signup-username').value;
    const email = document.getElementById('signup-email').value;
    const password = document.getElementById('signup-password').value;
    const errorDiv = document.getElementById('signup-error');
    
    try {
        const response = await fetch('/api/auth/signup', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                username, 
                email, 
                password,
                full_name: fullname
            })
        });
        
        if (response.ok) {
            const data = await response.json();
            localStorage.setItem('user_id', data.user_id);
            localStorage.setItem('username', data.username);
            window.location.href = '/static/chat.html';
        } else {
            const error = await response.json();
            errorDiv.textContent = error.detail || 'Signup failed';
        }
    } catch (error) {
        errorDiv.textContent = 'Connection error';
    }
});

// Enter key support
document.getElementById('login-password').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
        e.preventDefault();
        document.getElementById('login-form').requestSubmit();
    }
});

document.getElementById('signup-password').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
        e.preventDefault();
        document.getElementById('signup-form').requestSubmit();
    }
});


document.getElementById('signup-fullname').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
        e.preventDefault();
        document.getElementById('signup-username').focus();
    }
});