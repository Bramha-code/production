// Auth
const userId = localStorage.getItem('user_id');
if (!userId) window.location.href = '/';

let currentSessionId = null;
let websocket = null;
let sessionWebSockets = {};
let deleteConfirmSessionId = null;
let renameSessionId = null;
let uploadedFiles = [];
let isCreatingNewChat = false;
let newChatPending = false;

// Streaming control
let isStreaming = false;
let isPaused = false;
let currentStreamMessageId = null;
let stopRequested = false;  // Renamed from stopRequested for clarity
let lastUserMessageContent = null;

// Theme
const savedTheme = localStorage.getItem('theme') || 'light';
document.documentElement.setAttribute('data-theme', savedTheme);

// Selected model - Default to Gemini 2.5 Flash
// Valid models
const VALID_MODELS = ['gemini-2.5-flash', 'qwen2.5:7b'];
let savedModel = localStorage.getItem('selected_model');

// Clear invalid model from cache (handles old llama3.1:8b)
if (savedModel && !VALID_MODELS.includes(savedModel)) {
    console.log(`[MODEL] Clearing invalid cached model: ${savedModel}`);
    localStorage.removeItem('selected_model');
    savedModel = null;
}

let selectedModel = savedModel || 'gemini-2.5-flash';

// ============================================================================
// AUTO-SCROLL MANAGER (using MutationObserver for reliable scrolling)
// ============================================================================
class ScrollManager {
    constructor(containerSelector) {
        this.container = document.querySelector(containerSelector);
        if (!this.container) {
            console.error('ScrollManager: Container not found:', containerSelector);
            return;
        }

        this.shouldAutoScroll = true;
        this.isScrolling = false;
        this.initScrollDetection();
        this.initMutationObserver();
    }

    initScrollDetection() {
        // Detect when user manually scrolls
        this.container.addEventListener('scroll', () => {
            if (this.isScrolling) return; // Ignore programmatic scrolls

            // Check if user scrolled away from bottom
            const isNearBottom = this.container.scrollHeight - this.container.scrollTop
                <= this.container.clientHeight + 100;

            this.shouldAutoScroll = isNearBottom;
        });
    }

    initMutationObserver() {
        // Watch for content changes and auto-scroll if needed
        this.observer = new MutationObserver(() => {
            if (this.shouldAutoScroll) {
                this.scrollToBottom();
            }
        });

        this.observer.observe(this.container, {
            childList: true,       // Watch for added/removed messages
            subtree: true,         // Watch all descendants
            characterData: true    // Watch for text changes (streaming)
        });
    }

    scrollToBottom(smooth = false) {
        this.isScrolling = true;
        this.container.scrollTo({
            top: this.container.scrollHeight,
            behavior: smooth ? 'smooth' : 'instant'
        });

        // Reset flag after scroll completes
        setTimeout(() => { this.isScrolling = false; }, 100);
    }

    enable() {
        this.shouldAutoScroll = true;
        this.scrollToBottom();
    }

    disable() {
        this.shouldAutoScroll = false;
    }

    disconnect() {
        if (this.observer) {
            this.observer.disconnect();
        }
    }
}

// Global scroll manager instance
let chatScrollManager = null;

document.addEventListener('DOMContentLoaded', () => {
    // Initialize scroll manager
    chatScrollManager = new ScrollManager('#chat-messages');

    loadSessions();
    setupEventListeners();
    updateThemeIcon();
    updateSendButton();
    setupModelSelector();
    setupCharCounter();
    setupVoiceInput();
    initCodeHighlighting();
});

function setupEventListeners() {
    const newChatBtn = document.getElementById('new-chat-btn');
    if (newChatBtn) newChatBtn.addEventListener('click', createNewChat);

    const sendBtn = document.getElementById('send-btn');
    if (sendBtn) {
        sendBtn.addEventListener('click', () => {
            if (isStreaming) {
                if (isPaused) {
                    resumeStream();
                } else {
                    pauseStream();
                }
            } else {
                sendMessage();
            }
        });
    }

    const userInput = document.getElementById('user-input');
    if (userInput) {
        userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                if (isStreaming) {
                    if (isPaused) {
                        resumeStream();
                    } else {
                        pauseStream();
                    }
                } else {
                    sendMessage();
                }
            }
        });
        userInput.addEventListener('input', (e) => {
            autoResize(e.target);
            updateSendButton();
        });
    }

    // Enter key for rename modal
    const renameInput = document.getElementById('rename-input');
    if (renameInput) {
        renameInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                confirmRenameChat();
            }
        });
    }

    // Enter key for profile modal
    const profileEmail = document.getElementById('profile-email');
    const profileFullname = document.getElementById('profile-fullname');
    if (profileEmail) {
        profileEmail.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                saveProfile();
            }
        });
    }
    if (profileFullname) {
        profileFullname.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                saveProfile();
            }
        });
    }

    document.querySelectorAll('.quick-prompt').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const prompt = e.currentTarget.dataset.prompt;
            if (userInput) {
                userInput.value = prompt;
                updateSendButton();
                sendMessage();
            }
        });
    });

    const attachBtn = document.getElementById('attach-btn');
    if (attachBtn) {
        attachBtn.addEventListener('click', () => {
            document.getElementById('file-input').click();
        });
    }

    const fileInput = document.getElementById('file-input');
    if (fileInput) fileInput.addEventListener('change', handleFileUpload);

    const sidebarToggle = document.getElementById('sidebar-toggle');
    if (sidebarToggle) {
        sidebarToggle.addEventListener('click', () => {
            document.getElementById('sidebar').classList.toggle('collapsed');
        });
    }

    const searchInput = document.getElementById('search-input');
    if (searchInput) searchInput.addEventListener('input', handleSearch);

    const themeToggle = document.getElementById('theme-toggle');
    if (themeToggle) themeToggle.addEventListener('click', toggleTheme);

    const profileBtn = document.getElementById('profile-btn');
    if (profileBtn) profileBtn.addEventListener('click', showProfile);

    const faqBtn = document.getElementById('faq-btn');
    if (faqBtn) faqBtn.addEventListener('click', showFAQs);

    const saveProfileBtn = document.getElementById('save-profile-btn');
    if (saveProfileBtn) saveProfileBtn.addEventListener('click', saveProfile);

    const logoutBtn = document.getElementById('logout-btn');
    if (logoutBtn) logoutBtn.addEventListener('click', logout);


    document.querySelectorAll('.close-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.modal').forEach(m => m.classList.remove('active'));
        });
    });

    const confirmDelete = document.getElementById('confirm-delete');
    if (confirmDelete) confirmDelete.addEventListener('click', confirmDeleteChat);

    const cancelDelete = document.getElementById('cancel-delete');
    if (cancelDelete) {
        cancelDelete.addEventListener('click', () => {
            document.getElementById('delete-confirm').classList.remove('active');
        });
    }

    const confirmRename = document.getElementById('confirm-rename');
    if (confirmRename) confirmRename.addEventListener('click', confirmRenameChat);

    document.addEventListener('click', (e) => {
        if (!e.target.closest('.chat-item-menu') && !e.target.closest('.menu-dropdown')) {
            document.querySelectorAll('.menu-dropdown').forEach(d => d.remove());
        }
    });

    // ============================================================================
    // EVENT DELEGATION for message actions (copy, edit, regenerate, etc.)
    // ============================================================================
    document.addEventListener('click', (e) => {
        const target = e.target.closest('[data-action]');
        if (!target) return;

        const action = target.dataset.action;
        const msgElement = target.closest('.message');

        switch (action) {
            case 'copy':
                if (msgElement) copyMessage(msgElement);
                break;
            case 'copy-code':
                copyCodeBlock(target.dataset.codeId);
                break;
            case 'edit':
                if (msgElement) editMessage(msgElement);
                break;
            case 'regenerate':
                if (msgElement) regenerateResponse(msgElement);
                break;
            case 'cancel-edit':
                if (msgElement) cancelEdit(msgElement);
                break;
            case 'save-resend':
                if (msgElement) {
                    const textarea = msgElement.querySelector('.edit-message-input');
                    if (textarea) saveAndResend(msgElement, textarea.value);
                }
                break;
            case 'like':
                toggleLike(target);
                break;
            case 'dislike':
                toggleDislike(target);
                break;
            case 'share':
                if (msgElement) shareMessage(msgElement);
                break;
        }
    });
}

function toggleTheme() {
    const html = document.documentElement;
    const current = html.getAttribute('data-theme');
    const newTheme = current === 'light' ? 'dark' : 'light';
    html.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateThemeIcon();
}

function updateThemeIcon() {
    const icon = document.querySelector('#theme-toggle i');
    if (icon) {
        const theme = document.documentElement.getAttribute('data-theme');
        icon.className = theme === 'light' ? 'fas fa-moon' : 'fas fa-sun';
    }
}

function autoResize(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
}

function updateSendButton() {
    const input = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const sendIcon = sendBtn?.querySelector('i');

    if (!sendBtn || !sendIcon) return;

    if (isStreaming) {
        if (isPaused) {
            // Show resume button when paused
            sendBtn.className = 'resume-btn';
            sendBtn.title = 'Resume generation';
            sendBtn.disabled = false;
            sendIcon.className = 'fas fa-play';
        } else {
            // Show pause button during streaming
            sendBtn.className = 'pause-btn';
            sendBtn.title = 'Pause generation';
            sendBtn.disabled = false;
            sendIcon.className = 'fas fa-pause';
        }
    } else {
        // Show send button when not streaming
        sendBtn.className = 'send-btn';
        sendBtn.title = 'Send message';
        sendBtn.disabled = !input?.value.trim() && uploadedFiles.length === 0;
        sendIcon.className = 'fas fa-arrow-up';
    }
}

function setStreamingState(streaming, messageId = null) {
    isStreaming = streaming;
    currentStreamMessageId = messageId;
    updateSendButton();
}

async function interruptStream() {
    if (!isStreaming || !currentSessionId) return;

    console.log('🛑 Interrupting stream...');
    stopRequested = true;

    // Send interrupt signal via WebSocket
    if (websocket && websocket.readyState === WebSocket.OPEN) {
        websocket.send(JSON.stringify({ interrupt: true }));
        console.log('Interrupt signal sent');
    }

    // Immediately stop streaming state on client side
    setStreamingState(false);

    // Remove the assistant message completely
    const chatMessages = document.getElementById('chat-messages');
    if (chatMessages) {
        const currentMsg = document.querySelector(`[data-message-id="${currentStreamMessageId}"]`);
        if (currentMsg) {
            // Remove the entire assistant message
            currentMsg.remove();
            console.log('Assistant message removed from UI');
        }

        // Also remove the last user message if it matches the interrupted one
        const lastUserMsg = chatMessages.querySelector('.message.user:last-child');
        if (lastUserMsg && lastUserMessageContent) {
            const msgText = lastUserMsg.querySelector('.message-text');
            if (msgText && msgText.textContent === lastUserMessageContent) {
                lastUserMsg.remove();
                console.log('User message also removed from UI');
            }
        }
    }

    // Clear the last user message content
    lastUserMessageContent = null;
    console.log('Stream fully interrupted - nothing saved or displayed');
}

async function pauseStream() {
    if (!isStreaming || isPaused || !currentSessionId) return;

    console.log('⏸️ Pausing stream...');
    isPaused = true;
    updateSendButton();

    if (websocket && websocket.readyState === WebSocket.OPEN) {
        websocket.send(JSON.stringify({ action: 'pause' }));
        console.log('Pause signal sent');
    }
}

async function resumeStream() {
    if (!isStreaming || !isPaused || !currentSessionId) return;

    console.log('▶️ Resuming stream...');
    isPaused = false;
    updateSendButton();

    if (websocket && websocket.readyState === WebSocket.OPEN) {
        websocket.send(JSON.stringify({ action: 'resume' }));
        console.log('Resume signal sent');
    }

    // Process buffered chunks that arrived while paused
    if (window.pausedChunksBuffer && window.pausedChunksBuffer.length > 0) {
        console.log(`[DEBUG STREAM] Processing ${window.pausedChunksBuffer.length} buffered chunks`);
        const bufferedChunks = [...window.pausedChunksBuffer];
        window.pausedChunksBuffer = [];

        // Process each buffered chunk through the normal handler
        bufferedChunks.forEach(chunk => {
            handleWebSocketMessage(chunk);
        });
    }
}

async function stopGeneration() {
    if (!isStreaming || !currentSessionId) return;

    console.log('🛑 Stopping generation...');
    stopRequested = true;

    if (websocket && websocket.readyState === WebSocket.OPEN) {
        websocket.send(JSON.stringify({ action: 'stop' }));
        console.log('Stop signal sent');
    }

    setStreamingState(false);
    isPaused = false;
    updateSendButton();

    // Clear any buffered chunks
    if (window.pausedChunksBuffer) {
        window.pausedChunksBuffer = [];
        console.log('Cleared buffered chunks');
    }

    // Remove the partial assistant message
    const chatMessages = document.getElementById('chat-messages');
    if (chatMessages && currentStreamMessageId) {
        const currentMsg = document.querySelector(`[data-message-id="${currentStreamMessageId}"]`);
        if (currentMsg) {
            currentMsg.remove();
            console.log('Partial assistant message removed');
        }
    }
}

// ============================================================================
// TOAST NOTIFICATIONS
// ============================================================================
function showToast(message, type = 'info', duration = 3000) {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);

    // Show toast with animation
    setTimeout(() => toast.classList.add('show'), 10);

    // Auto-remove after duration
    if (duration > 0) {
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300); // Wait for fade-out animation
        }, duration);
    }
}

async function loadSessions() {
    try {
        const response = await fetch(`/api/sessions/${userId}`);
        const data = await response.json();
        renderSessions(data.sessions);
    } catch (error) {
        console.error('Error loading sessions:', error);
    }
}

function renderSessions(sessions) {
    const list = document.getElementById('chat-list');
    if (!list) return;
    list.innerHTML = '';

    const starred = sessions.filter(s => s.starred);
    const recent = sessions.filter(s => !s.starred);

    // Starred section
    if (starred.length > 0) {
        const starredHeader = document.createElement('div');
        starredHeader.className = 'chat-section-header';
        starredHeader.textContent = 'Starred';
        list.appendChild(starredHeader);

        starred.forEach(session => {
            list.appendChild(createChatItem(session));
        });
    }

    // Recent section
    if (recent.length > 0) {
        const recentHeader = document.createElement('div');
        recentHeader.className = 'chat-section-header';
        recentHeader.textContent = 'Recent';
        list.appendChild(recentHeader);

        recent.forEach(session => {
            list.appendChild(createChatItem(session));
        });
    }
}

function createChatItem(session) {
    const item = document.createElement('div');
    item.className = 'chat-item';
    if (session.session_id === currentSessionId) {
        item.classList.add('active');
    }

    item.innerHTML = `
        <div class="chat-item-content">
            <div class="chat-item-title">${escapeHtml(session.title)}</div>
            <div class="chat-item-time">${formatTime(session.updated_at)}</div>
        </div>
        <button class="menu-btn more-btn" data-id="${session.session_id}">
            <i class="fas fa-ellipsis-v"></i>
        </button>
    `;

    const contentDiv = item.querySelector('.chat-item-content');
    contentDiv.addEventListener('click', () => {
        loadSession(session.session_id);
    });

    const moreBtn = item.querySelector('.more-btn');
    moreBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        showChatMenu(e, session);
    });

    return item;
}

function showChatMenu(event, session) {
    document.querySelectorAll('.menu-dropdown').forEach(d => d.remove());

    const dropdown = document.createElement('div');
    dropdown.className = 'menu-dropdown';
    dropdown.innerHTML = `
        <button class="dropdown-item star-item">
            <i class="fas fa-star ${session.starred ? 'starred' : ''}"></i>
            ${session.starred ? 'Unstar' : 'Star'}
        </button>
        <button class="dropdown-item rename-item">
            <i class="fas fa-edit"></i>
            Rename
        </button>
        <button class="dropdown-item delete-item">
            <i class="fas fa-trash"></i>
            Delete
        </button>
    `;

    const rect = event.target.closest('.menu-btn').getBoundingClientRect();
    dropdown.style.position = 'absolute';
    dropdown.style.left = (rect.left - 120) + 'px';
    dropdown.style.top = (rect.bottom + 5) + 'px';
    dropdown.style.zIndex = '1000';

    document.body.appendChild(dropdown);

    dropdown.querySelector('.star-item').addEventListener('click', async () => {
        await starSession(session.session_id);
        dropdown.remove();
    });

    dropdown.querySelector('.rename-item').addEventListener('click', () => {
        showRenameDialog(session.session_id, session.title);
        dropdown.remove();
    });

    dropdown.querySelector('.delete-item').addEventListener('click', () => {
        showDeleteConfirm(session.session_id);
        dropdown.remove();
    });
}

async function createNewChat() {
    if (isCreatingNewChat || newChatPending) return;

    try {
        isCreatingNewChat = true;
        newChatPending = true;

        const response = await fetch(`/api/sessions/${userId}`, { method: 'POST' });
        const data = await response.json();
        currentSessionId = data.session_id;

        const chatMessages = document.getElementById('chat-messages');
        if (chatMessages) chatMessages.innerHTML = '';

        const welcomeScreen = document.getElementById('welcome-screen');
        if (welcomeScreen) welcomeScreen.style.display = 'flex';

        const messagesContainer = document.getElementById('messages-container');
        if (messagesContainer) messagesContainer.classList.add('centered');

        const chatTitle = document.getElementById('current-chat-title');
        if (chatTitle) chatTitle.textContent = 'New Chat';

        connectWebSocket();
        await loadSessions();
    } catch (error) {
        console.error('Error creating session:', error);
    } finally {
        isCreatingNewChat = false;
    }
}

async function loadSession(sessionId) {
    try {
        const response = await fetch(`/api/session/${sessionId}`);
        const session = await response.json();

        currentSessionId = sessionId;
        newChatPending = false;

        const welcomeScreen = document.getElementById('welcome-screen');
        if (welcomeScreen) welcomeScreen.style.display = 'none';

        const messagesContainer = document.getElementById('messages-container');
        if (messagesContainer) messagesContainer.classList.remove('centered');

        const chatTitle = document.getElementById('current-chat-title');
        if (chatTitle) chatTitle.textContent = session.title;

        const chatMessages = document.getElementById('chat-messages');
        if (chatMessages) {
            chatMessages.innerHTML = '';
            session.messages.forEach(msg => displayMessage(msg));

            // Use scroll manager for reliable scrolling
            if (chatScrollManager) {
                chatScrollManager.enable();
            }
        }

        connectWebSocket();
        await loadSessions();
    } catch (error) {
        console.error('Error loading session:', error);
    }
}

async function starSession(sessionId) {
    try {
        await fetch(`/api/session/${sessionId}/star`, { method: 'POST' });
        await loadSessions();
    } catch (error) {
        console.error('Error starring session:', error);
    }
}

function showDeleteConfirm(sessionId) {
    deleteConfirmSessionId = sessionId;
    const deleteConfirm = document.getElementById('delete-confirm');
    if (deleteConfirm) {
        deleteConfirm.classList.add('active');
        // Position near sidebar
        deleteConfirm.style.position = 'fixed';
        deleteConfirm.style.left = '350px';
        deleteConfirm.style.top = '50%';
        deleteConfirm.style.transform = 'translateY(-50%)';
    }
}

async function confirmDeleteChat() {
    if (!deleteConfirmSessionId) return;

    try {
        await fetch(`/api/session/${deleteConfirmSessionId}`, { method: 'DELETE' });

        if (deleteConfirmSessionId === currentSessionId) {
            currentSessionId = null;
            newChatPending = false;

            const chatMessages = document.getElementById('chat-messages');
            if (chatMessages) chatMessages.innerHTML = '';

            const welcomeScreen = document.getElementById('welcome-screen');
            if (welcomeScreen) welcomeScreen.style.display = 'flex';

            const messagesContainer = document.getElementById('messages-container');
            if (messagesContainer) messagesContainer.classList.add('centered');

            const chatTitle = document.getElementById('current-chat-title');
            if (chatTitle) chatTitle.textContent = 'New Chat';
        }

        const deleteConfirm = document.getElementById('delete-confirm');
        if (deleteConfirm) deleteConfirm.classList.remove('active');

        await loadSessions();
    } catch (error) {
        console.error('Error deleting session:', error);
    }
}

function showRenameDialog(sessionId, currentTitle) {
    renameSessionId = sessionId;
    const renameInput = document.getElementById('rename-input');
    if (renameInput) {
        renameInput.value = currentTitle;
        renameInput.focus();
        renameInput.select();
    }

    const renameModal = document.getElementById('rename-modal');
    if (renameModal) renameModal.classList.add('active');
}

async function confirmRenameChat() {
    const renameInput = document.getElementById('rename-input');
    const newTitle = renameInput ? renameInput.value.trim() : '';
    if (!newTitle || !renameSessionId) return;

    try {
        await fetch(`/api/session/${renameSessionId}/rename`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ new_title: newTitle })
        });

        if (renameSessionId === currentSessionId) {
            const chatTitle = document.getElementById('current-chat-title');
            if (chatTitle) chatTitle.textContent = newTitle;
        }

        const renameModal = document.getElementById('rename-modal');
        if (renameModal) renameModal.classList.remove('active');

        await loadSessions();
    } catch (error) {
        console.error('Error renaming session:', error);
    }
}

async function handleSearch(e) {
    const query = e.target.value.trim();

    if (!query) {
        loadSessions();
        return;
    }

    try {
        const response = await fetch(`/api/search/${userId}?q=${encodeURIComponent(query)}`);
        const data = await response.json();
        renderSessions(data.sessions);
    } catch (error) {
        console.error('Error searching:', error);
    }
}

function updateConnectionStatus(status) {
    const statusEl = document.getElementById('connection-status');
    if (!statusEl) return;

    const statusDot = statusEl.querySelector('.status-dot');
    const statusText = statusEl.querySelector('.status-text');

    statusEl.className = 'connection-status ' + status;

    switch (status) {
        case 'connected':
            statusText.textContent = 'Connected';
            statusEl.style.display = 'flex';
            // Hide after 3 seconds when connected
            setTimeout(() => {
                if (statusEl.classList.contains('connected')) {
                    statusEl.style.display = 'none';
                }
            }, 3000);
            break;
        case 'connecting':
            statusText.textContent = 'Connecting...';
            statusEl.style.display = 'flex';
            break;
        case 'disconnected':
            statusText.textContent = 'Disconnected';
            statusEl.style.display = 'flex';
            break;
    }
}

function connectWebSocket() {
    if (!currentSessionId) {
        console.log('No session ID, skipping WebSocket connection');
        return;
    }

    if (sessionWebSockets[currentSessionId]) {
        websocket = sessionWebSockets[currentSessionId];
        console.log('Reusing existing WebSocket for session:', currentSessionId);
        if (websocket.readyState === WebSocket.OPEN) {
            updateConnectionStatus('connected');
        }
        return;
    }

    updateConnectionStatus('connecting');

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/chat/${currentSessionId}`;
    console.log('Connecting WebSocket to:', wsUrl);

    websocket = new WebSocket(wsUrl);
    sessionWebSockets[currentSessionId] = websocket;

    websocket.onopen = () => {
        console.log('WebSocket connected successfully');
        updateConnectionStatus('connected');
        stopRequested = false;
        lastUserMessageContent = null;
    };

    websocket.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);

            // Check if interrupt was requested - ignore ALL incoming messages
            if (stopRequested) {
                console.log('Ignoring ALL WebSocket messages after interrupt:', data.type);
                return;
            }

            console.log('WebSocket message received:', data.type);
            handleWebSocketMessage(data);
        } catch (error) {
            console.error('Error parsing WebSocket message:', error, event.data);
        }
    };

    websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
        updateConnectionStatus('disconnected');
        setStreamingState(false);
        stopRequested = false;
    };

    websocket.onclose = (event) => {
        console.log('WebSocket closed:', event.code, event.reason);
        delete sessionWebSockets[currentSessionId];
        updateConnectionStatus('disconnected');
        setStreamingState(false);
        stopRequested = false;
        lastUserMessageContent = null;
    };
}

function handleWebSocketMessage(data) {
    const chatMessages = document.getElementById('chat-messages');
    if (!chatMessages) {
        console.error('chat-messages container not found');
        return;
    }

    // Handle pause/resume/stop actions
    if (data.type === 'paused') {
        console.log('⏸️ Stream paused by server');
        return;
    }

    if (data.type === 'resumed') {
        console.log('▶️ Stream resumed by server');
        return;
    }

    if (data.type === 'stopped') {
        console.log('🛑 Stream stopped by server');
        setStreamingState(false);
        isPaused = false;
        updateSendButton();
        return;
    }

    if (data.type === 'metadata') {
        console.log('Creating assistant message container for:', data.data.message_id);
        setStreamingState(true, data.data.message_id);

        // Create assistant message element
        const assistantMsg = document.createElement('div');
        assistantMsg.className = 'message assistant';
        assistantMsg.setAttribute('data-message-id', data.data.message_id);
        assistantMsg.setAttribute('id', `msg-${data.data.message_id}`);

        // Add avatar
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = '<i class="fas fa-robot"></i>';
        assistantMsg.appendChild(avatar);

        const msgContent = document.createElement('div');
        msgContent.className = 'message-content';

        const msgText = document.createElement('div');
        msgText.className = 'message-text';
        msgText.innerHTML = '<div class="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div>';
        msgText.dataset.rawText = '';

        msgContent.appendChild(msgText);
        assistantMsg.appendChild(msgContent);
        chatMessages.appendChild(assistantMsg);

        console.log('Assistant message container created');

    } else if (data.type === 'stream') {
        const messageId = data.data.message_id;
        console.log('[DEBUG STREAM] Received chunk:', data.data.content, 'for message:', messageId);

        // CHECK PAUSE STATE - Buffer chunks if paused
        if (isPaused) {
            console.log('[DEBUG STREAM] Stream is PAUSED - buffering chunk');
            // Store chunk in a buffer for later display when resumed
            if (!window.pausedChunksBuffer) {
                window.pausedChunksBuffer = [];
            }
            window.pausedChunksBuffer.push(data);
            return; // Don't process the chunk yet
        }

        let assistantMsg = document.querySelector(`[data-message-id="${messageId}"]`);

        if (!assistantMsg) {
            console.log('Message element not found, creating new one for:', messageId);
            assistantMsg = document.createElement('div');
            assistantMsg.className = 'message assistant';
            assistantMsg.setAttribute('data-message-id', messageId);
            assistantMsg.setAttribute('id', `msg-${messageId}`);

            // Add avatar
            const avatar = document.createElement('div');
            avatar.className = 'message-avatar';
            avatar.innerHTML = '<i class="fas fa-robot"></i>';
            assistantMsg.appendChild(avatar);

            const msgContent = document.createElement('div');
            msgContent.className = 'message-content';

            const msgText = document.createElement('div');
            msgText.className = 'message-text';
            msgText.dataset.rawText = '';

            msgContent.appendChild(msgText);
            assistantMsg.appendChild(msgContent);
            chatMessages.appendChild(assistantMsg);
        }

        const msgText = assistantMsg.querySelector('.message-text');
        if (!msgText) {
            console.error('Message text element not found');
            return;
        }

        // Remove typing indicator if present
        if (msgText.querySelector('.typing-indicator')) {
            msgText.innerHTML = '';
            msgText.dataset.rawText = '';
        }

        // Append new content
        if (!msgText.dataset.rawText) msgText.dataset.rawText = '';
        msgText.dataset.rawText += data.data.content;

        console.log('[DEBUG STREAM] Total accumulated text:', msgText.dataset.rawText.length, 'chars');
        console.log('[DEBUG STREAM] First 100 chars:', msgText.dataset.rawText.substring(0, 100));

        // Update display with formatted markdown
        msgText.innerHTML = formatMarkdown(msgText.dataset.rawText);
        renderEnhancements(msgText, true); // Skip Mermaid during streaming to prevent flickering
        console.log('[DEBUG STREAM] Formatted HTML length:', msgText.innerHTML.length);

        // Scroll to show new content
        setTimeout(() => {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }, 50);

    } else if (data.type === 'complete') {
        console.log('Stream complete for message:', data.data.message_id);
        const messageId = data.data.message_id;
        const assistantMsg = document.querySelector(`[data-message-id="${messageId}"]`);

        if (assistantMsg) {
            const msgText = assistantMsg.querySelector('.message-text');
            if (msgText && msgText.dataset.rawText) {
                // Final formatting
                msgText.innerHTML = formatMarkdown(msgText.dataset.rawText);
                renderEnhancements(msgText, false); // Render Mermaid diagrams ONLY on completion

                // Check if this is a test plan response using backend flag
                const isTestPlan = data.data.is_test_plan === true;
                const exportAvailable = data.data.export_available === true;
                const testPlanQuery = data.data.test_plan_query || lastUserMessageContent;
                const downloadAvailable = data.data.download_available === true;
                const downloadUrl = data.data.download_url;
                const fileName = data.data.file_name;

                if (isTestPlan && exportAvailable && testPlanQuery) {
                    setTimeout(() => {
                        addExportButton(assistantMsg, testPlanQuery, downloadAvailable, downloadUrl, fileName, messageId);
                    }, 100);
                }

                // Display images from knowledge graph if available
                if (data.data.images && data.data.images.length > 0) {
                    console.log('[IMAGES] Displaying', data.data.images.length, 'images from knowledge graph');
                    const msgContent = assistantMsg.querySelector('.message-content');
                    if (msgContent && !msgContent.querySelector('.image-gallery')) {
                        displayImagesInMessage(msgContent, data.data.images);
                    }
                }

                // Keep rawText for export
                // delete msgText.dataset.rawText;
            }

            // ADD ACTION BUTTONS after streaming completes
            const msgContent = assistantMsg.querySelector('.message-content');
            if (msgContent && !msgContent.querySelector('.message-actions')) {
                const actions = document.createElement('div');
                actions.className = 'message-actions';
                actions.setAttribute('role', 'toolbar');
                actions.setAttribute('aria-label', 'Message actions');

                // Modern SVG icons (Gemini-style)
                const iconButtons = [
                    {
                        name: 'copy',
                        title: 'Copy',
                        ariaLabel: 'Copy message to clipboard',
                        svg: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>'
                    },
                    {
                        name: 'like',
                        title: 'Good response',
                        ariaLabel: 'Mark response as helpful',
                        svg: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"></path></svg>'
                    },
                    {
                        name: 'dislike',
                        title: 'Bad response',
                        ariaLabel: 'Mark response as not helpful',
                        svg: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 1-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"></path></svg>'
                    },
                    {
                        name: 'share',
                        title: 'Share',
                        ariaLabel: 'Share message',
                        svg: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="18" cy="5" r="3"></circle><circle cx="6" cy="12" r="3"></circle><circle cx="18" cy="19" r="3"></circle><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"></line><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"></line></svg>'
                    },
                    {
                        name: 'regenerate',
                        title: 'Regenerate',
                        ariaLabel: 'Regenerate response',
                        svg: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"></polyline><polyline points="1 20 1 14 7 14"></polyline><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path></svg>'
                    }
                ];

                iconButtons.forEach(icon => {
                    const btn = document.createElement('button');
                    btn.className = `action-btn ${icon.name}-btn`;
                    btn.innerHTML = icon.svg;
                    btn.title = icon.title;
                    btn.setAttribute('aria-label', icon.ariaLabel);
                    btn.setAttribute('data-action', icon.name);
                    btn.setAttribute('tabindex', '0');
                    actions.appendChild(btn);
                });

                msgContent.appendChild(actions);
                console.log('Action buttons added to completed message');
            }
        }

        // Update session list
        newChatPending = false;
        loadSessions();

        // Final scroll
        setTimeout(() => {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }, 100);

        setStreamingState(false);
        stopRequested = false;
        lastUserMessageContent = null;

    } else if (data.type === 'error') {
        console.error('WebSocket error:', data.data);
        const lastMsg = chatMessages.querySelector('.message.assistant:last-child .message-text');
        if (lastMsg) {
            lastMsg.innerHTML = '<p style="color: #ef4444;">Error: ' + escapeHtml(data.data.message || 'Unknown error') + '</p>';
        }
        setStreamingState(false);
        stopRequested = false;
        lastUserMessageContent = null;
    }
}

// =========================================================
// ChatGPT-like Features
// =========================================================

// Model Selector
function setupModelSelector() {
    const modelBtn = document.getElementById('model-dropdown-btn');
    const modelSelector = document.querySelector('.model-selector');
    const modelOptions = document.querySelectorAll('.model-option');

    if (!modelBtn || !modelSelector) return;

    // Set initial model display
    const savedModel = localStorage.getItem('selected_model') || 'gemini-2.5-flash';
    updateModelDisplay(savedModel);

    // Toggle dropdown
    modelBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        modelSelector.classList.toggle('open');
    });

    // Select model
    modelOptions.forEach(option => {
        option.addEventListener('click', () => {
            const model = option.dataset.model;
            selectedModel = model;
            localStorage.setItem('selected_model', model);

            // Update active state
            modelOptions.forEach(o => o.classList.remove('active'));
            option.classList.add('active');

            updateModelDisplay(model);
            modelSelector.classList.remove('open');
        });
    });

    // Close dropdown when clicking outside
    document.addEventListener('click', (e) => {
        if (!modelSelector.contains(e.target)) {
            modelSelector.classList.remove('open');
        }
    });
}

function updateModelDisplay(model) {
    const currentModelName = document.getElementById('current-model-name');
    if (!currentModelName) return;

    const modelNames = {
        'qwen2.5:7b': 'Qwen 2.5 7B',
        'llama3.1:8b': 'Llama 3.1',
        'gemini-2.5-flash': 'Gemini 2.5 Flash'
    };
    currentModelName.textContent = modelNames[model] || model;
}

// Character Counter
function setupCharCounter() {
    const input = document.getElementById('user-input');
    const counter = document.getElementById('char-counter');

    if (!input || !counter) return;

    const maxChars = 4000;

    input.addEventListener('input', () => {
        const len = input.value.length;
        counter.textContent = `${len} / ${maxChars}`;

        counter.classList.remove('warning', 'danger');
        if (len > maxChars * 0.9) {
            counter.classList.add('danger');
        } else if (len > maxChars * 0.75) {
            counter.classList.add('warning');
        }
    });
}

// Voice Input
let recognition = null;
let isRecording = false;

function setupVoiceInput() {
    const voiceBtn = document.getElementById('voice-btn');
    if (!voiceBtn) return;

    // Check for speech recognition support
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
        voiceBtn.style.display = 'none';
        return;
    }

    recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    recognition.onresult = (event) => {
        const input = document.getElementById('user-input');
        if (!input) return;

        let transcript = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
            transcript += event.results[i][0].transcript;
        }
        input.value = transcript;
        autoResize(input);
        updateSendButton();

        // Update character counter
        const counter = document.getElementById('char-counter');
        if (counter) {
            counter.textContent = `${input.value.length} / 4000`;
        }
    };

    recognition.onend = () => {
        isRecording = false;
        voiceBtn.classList.remove('recording');
        voiceBtn.innerHTML = '<i class="fas fa-microphone"></i>';
    };

    recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        isRecording = false;
        voiceBtn.classList.remove('recording');
        voiceBtn.innerHTML = '<i class="fas fa-microphone"></i>';
    };

    voiceBtn.addEventListener('click', toggleVoiceInput);
}

function toggleVoiceInput() {
    const voiceBtn = document.getElementById('voice-btn');
    if (!voiceBtn || !recognition) return;

    if (isRecording) {
        recognition.stop();
        isRecording = false;
        voiceBtn.classList.remove('recording');
        voiceBtn.innerHTML = '<i class="fas fa-microphone"></i>';
    } else {
        recognition.start();
        isRecording = true;
        voiceBtn.classList.add('recording');
        voiceBtn.innerHTML = '<i class="fas fa-stop"></i>';
    }
}

// Code Syntax Highlighting
function initCodeHighlighting() {
    if (typeof hljs !== 'undefined') {
        hljs.configure({
            ignoreUnescapedHTML: true
        });
    }
}

// Copy code block
function copyCodeBlock(codeId) {
    const codeElement = document.getElementById(codeId);
    if (!codeElement) return;

    const text = codeElement.textContent;
    navigator.clipboard.writeText(text).then(() => {
        const wrapper = codeElement.closest('.code-block-wrapper');
        const btn = wrapper?.querySelector('.copy-code-btn');
        if (btn) {
            const originalHtml = btn.innerHTML;
            btn.innerHTML = '<i class="fas fa-check"></i> Copied!';
            btn.classList.add('copied');
            setTimeout(() => {
                btn.innerHTML = originalHtml;
                btn.classList.remove('copied');
            }, 2000);
        }
    }).catch(err => {
        console.error('Failed to copy code:', err);
    });
}

// Like/Dislike toggle functions
function toggleLike(btn) {
    const dislikeBtn = btn.parentElement.querySelector('.dislike-btn');

    if (btn.classList.contains('liked')) {
        btn.classList.remove('liked');
        btn.innerHTML = '<i class="far fa-thumbs-up"></i>';
    } else {
        btn.classList.add('liked');
        btn.innerHTML = '<i class="fas fa-thumbs-up"></i>';

        if (dislikeBtn && dislikeBtn.classList.contains('disliked')) {
            dislikeBtn.classList.remove('disliked');
            dislikeBtn.innerHTML = '<i class="far fa-thumbs-down"></i>';
        }
    }
}

function toggleDislike(btn) {
    const likeBtn = btn.parentElement.querySelector('.like-btn');

    if (btn.classList.contains('disliked')) {
        btn.classList.remove('disliked');
        btn.innerHTML = '<i class="far fa-thumbs-down"></i>';
    } else {
        btn.classList.add('disliked');
        btn.innerHTML = '<i class="fas fa-thumbs-down"></i>';

        if (likeBtn && likeBtn.classList.contains('liked')) {
            likeBtn.classList.remove('liked');
            likeBtn.innerHTML = '<i class="far fa-thumbs-up"></i>';
        }
    }
}

function shareMessage(msgElement) {
    const msgText = msgElement.querySelector('.message-text');
    if (!msgText) return;

    // Get plain text content
    const text = msgText.innerText || msgText.textContent;

    // Check if Web Share API is available
    if (navigator.share) {
        navigator.share({
            title: 'Shared from Millennium Techlink',
            text: text
        }).then(() => {
            showToast('Message shared successfully', 'success', 2000);
        }).catch(err => {
            if (err.name !== 'AbortError') {
                console.error('Share failed:', err);
                fallbackShare(text);
            }
        });
    } else {
        // Fallback: Copy to clipboard
        fallbackShare(text);
    }
}

function fallbackShare(text) {
    navigator.clipboard.writeText(text).then(() => {
        showToast('Message copied to clipboard (share fallback)', 'success', 2000);
    }).catch(err => {
        console.error('Failed to copy:', err);
        showToast('Failed to share message', 'error');
    });
}

async function sendMessage() {
    const input = document.getElementById('user-input');
    const message = input ? input.value.trim() : '';

    if (!message && uploadedFiles.length === 0) return;

    // Store the message content for potential removal if interrupted
    lastUserMessageContent = message;

    if (!currentSessionId || newChatPending) {
        await createNewChat();
        await new Promise(resolve => setTimeout(resolve, 500));
    }

    const isFirstMessage = document.getElementById('chat-messages')?.children.length === 0;
    if (isFirstMessage && message) {
        const newTitle = message.substring(0, 50).trim();
        try {
            await fetch(`/api/session/${currentSessionId}/rename`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ new_title: newTitle })
            });

            const chatTitle = document.getElementById('current-chat-title');
            if (chatTitle) chatTitle.textContent = newTitle;

            loadSessions();
        } catch (error) {
            console.error('Error renaming chat:', error);
        }
    }

    const welcomeScreen = document.getElementById('welcome-screen');
    if (welcomeScreen) welcomeScreen.style.display = 'none';

    const messagesContainer = document.getElementById('messages-container');
    if (messagesContainer) messagesContainer.classList.remove('centered');

    const chatMessages = document.getElementById('chat-messages');
    if (chatMessages) {
        const userMsg = createMessageElement('user', message);

        if (uploadedFiles.length > 0) {
            const attachmentsDiv = document.createElement('div');
            attachmentsDiv.className = 'message-attachments';

            uploadedFiles.forEach(file => {
                const attachment = document.createElement('div');
                const isImage = file.filename.match(/\.(jpg|jpeg|png|gif|webp)$/i);

                if (isImage) {
                    attachment.className = 'attachment-image';
                    attachment.innerHTML = `<img src="/api/file/${userId}/${file.file_id}" alt="${escapeHtml(file.filename)}">`;
                } else {
                    attachment.className = 'attachment-doc';
                    attachment.innerHTML = `
                        <i class="fas fa-file-alt"></i>
                        <span>${escapeHtml(file.filename)}</span>
                    `;
                }

                attachmentsDiv.appendChild(attachment);
            });

            const msgContent = userMsg.querySelector('.message-content');
            msgContent.insertBefore(attachmentsDiv, msgContent.firstChild);
        }

        chatMessages.appendChild(userMsg);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    if (input) {
        input.value = '';
        autoResize(input);
    }
    updateSendButton();

    const filesData = [...uploadedFiles];
    uploadedFiles = [];
    const filePreview = document.getElementById('file-preview');
    if (filePreview) filePreview.innerHTML = '';

    const messageWithFiles = {
        message: message,
        model: selectedModel,  // Send selected model to backend
        files: filesData.map(f => ({
            filename: f.filename,
            file_id: f.file_id,
            temp_path: f.path
        }))
    };

    console.log('[DEBUG] Sending message with model:', selectedModel);

    if (websocket && websocket.readyState === WebSocket.OPEN) {
        websocket.send(JSON.stringify(messageWithFiles));
        console.log('Message sent via WebSocket');
    } else {
        console.log('WebSocket not open, connecting...');
        connectWebSocket();
        setTimeout(() => {
            if (websocket && websocket.readyState === WebSocket.OPEN) {
                websocket.send(JSON.stringify(messageWithFiles));
                console.log('Message sent after reconnection');
            } else {
                console.error('WebSocket still not open');
            }
        }, 1000);
    }
}

function createMessageElement(role, content, timestamp = null) {
    const msg = document.createElement('div');
    msg.className = `message ${role}`;
    const messageId = 'msg-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
    msg.setAttribute('data-message-id', messageId);

    // Add avatar (simple circles like ChatGPT)
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    if (role === 'user') {
        // User avatar - initials in a circle
        avatar.textContent = 'U';
    } else {
        // Assistant avatar - simple ChatGPT logo style
        avatar.innerHTML = '<svg width="24" height="24" viewBox="0 0 24 24" fill="none"><path d="M12 2L2 7L12 12L22 7L12 2Z" fill="currentColor"/><path d="M2 17L12 22L22 17M2 12L12 17L22 12" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>';
    }
    msg.appendChild(avatar);

    const msgContent = document.createElement('div');
    msgContent.className = 'message-content';

    if (content) {
        const msgText = document.createElement('div');
        msgText.className = 'message-text';
        if (role === 'assistant') {
            msgText.innerHTML = formatMarkdown(content);
            renderEnhancements(msgText); // Render mermaid diagrams, math, etc.
        } else {
            msgText.textContent = content;
        }
        msgContent.appendChild(msgText);
    }

    // Add action buttons ONLY for assistant messages, positioned at bottom-right
    if (role === 'assistant') {
        const actions = document.createElement('div');
        actions.className = 'message-actions';
        actions.setAttribute('role', 'toolbar');
        actions.setAttribute('aria-label', 'Message actions');

        // Modern SVG icons (Gemini-style)
        const iconButtons = [
            {
                name: 'copy',
                title: 'Copy',
                ariaLabel: 'Copy message to clipboard',
                svg: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>'
            },
            {
                name: 'like',
                title: 'Good response',
                ariaLabel: 'Mark response as helpful',
                svg: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"></path></svg>'
            },
            {
                name: 'dislike',
                title: 'Bad response',
                ariaLabel: 'Mark response as not helpful',
                svg: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 1-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"></path></svg>'
            },
            {
                name: 'share',
                title: 'Share',
                ariaLabel: 'Share message',
                svg: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="18" cy="5" r="3"></circle><circle cx="6" cy="12" r="3"></circle><circle cx="18" cy="19" r="3"></circle><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"></line><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"></line></svg>'
            },
            {
                name: 'regenerate',
                title: 'Regenerate',
                ariaLabel: 'Regenerate response',
                svg: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"></polyline><polyline points="1 20 1 14 7 14"></polyline><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path></svg>'
            }
        ];

        iconButtons.forEach(icon => {
            const btn = document.createElement('button');
            btn.className = `action-btn ${icon.name}-btn`;
            btn.innerHTML = icon.svg;
            btn.title = icon.title;
            btn.setAttribute('aria-label', icon.ariaLabel);
            btn.setAttribute('data-action', icon.name);
            btn.setAttribute('tabindex', '0');
            actions.appendChild(btn);
        });

        msgContent.appendChild(actions);
    }

    // For user messages, add minimal edit action
    if (role === 'user') {
        const actions = document.createElement('div');
        actions.className = 'message-actions user-actions';
        actions.setAttribute('role', 'toolbar');
        actions.setAttribute('aria-label', 'Message actions');

        const editBtn = document.createElement('button');
        editBtn.className = 'action-btn edit-btn';
        editBtn.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path></svg>';
        editBtn.title = 'Edit message';
        editBtn.setAttribute('aria-label', 'Edit message');
        editBtn.setAttribute('data-action', 'edit');
        editBtn.setAttribute('tabindex', '0');
        actions.appendChild(editBtn);

        msgContent.appendChild(actions);
    }

    msg.appendChild(msgContent);

    return msg;
}

function formatMessageTime(date) {
    const now = new Date();
    const isToday = date.toDateString() === now.toDateString();

    const timeStr = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    if (isToday) {
        return timeStr;
    } else {
        return date.toLocaleDateString([], { month: 'short', day: 'numeric' }) + ' ' + timeStr;
    }
}

function copyMessage(msgElement) {
    const msgText = msgElement.querySelector('.message-text');
    if (!msgText) return;

    // Get plain text content
    const text = msgText.innerText || msgText.textContent;

    navigator.clipboard.writeText(text).then(() => {
        // Show toast notification
        showToast('Message copied to clipboard', 'success', 2000);

        // Show button feedback
        const copyBtn = msgElement.querySelector('.copy-btn');
        if (copyBtn) {
            const originalHtml = copyBtn.innerHTML;
            copyBtn.innerHTML = '<i class="fas fa-check"></i>';
            copyBtn.classList.add('copied');
            setTimeout(() => {
                copyBtn.innerHTML = originalHtml;
                copyBtn.classList.remove('copied');
            }, 1500);
        }
    }).catch(err => {
        console.error('Failed to copy:', err);
        showToast('Failed to copy message', 'error');
    });
}

async function regenerateResponse(msgElement) {
    if (isStreaming) return;

    const chatMessages = document.getElementById('chat-messages');
    if (!chatMessages) return;

    // Find the previous user message
    const allMessages = Array.from(chatMessages.querySelectorAll('.message'));
    const msgIndex = allMessages.indexOf(msgElement);

    if (msgIndex <= 0) return;

    // Find the last user message before this assistant message
    let userMessage = null;
    for (let i = msgIndex - 1; i >= 0; i--) {
        if (allMessages[i].classList.contains('user')) {
            userMessage = allMessages[i];
            break;
        }
    }

    if (!userMessage) return;

    const userText = userMessage.querySelector('.message-text');
    if (!userText) return;

    const originalQuery = userText.textContent;

    // Remove the current assistant message
    msgElement.remove();

    // Store the message for potential interrupt removal
    lastUserMessageContent = originalQuery;

    // Send the message again via WebSocket
    if (websocket && websocket.readyState === WebSocket.OPEN) {
        websocket.send(JSON.stringify({
            message: originalQuery,
            model: selectedModel,  // Send selected model to backend
            files: [],
            regenerate: true
        }));
        console.log('Regenerate request sent');
    }
}

// ============================================================================
// EDIT & RESEND FUNCTIONALITY
// ============================================================================
function editMessage(msgElement) {
    if (isStreaming) {
        showToast('Cannot edit while streaming', 'error');
        return;
    }

    const msgText = msgElement.querySelector('.message-text');
    const msgFooter = msgElement.querySelector('.message-footer');
    if (!msgText) return;

    const originalText = msgText.innerText || msgText.textContent;

    // Create edit UI
    const editContainer = document.createElement('div');
    editContainer.className = 'edit-message-container';

    const textarea = document.createElement('textarea');
    textarea.className = 'edit-message-input';
    textarea.value = originalText;
    textarea.rows = Math.min(originalText.split('\n').length + 1, 10);

    const editActions = document.createElement('div');
    editActions.className = 'edit-message-actions';
    editActions.innerHTML = `
        <button class="btn-cancel" data-action="cancel-edit">
            <i class="fas fa-times"></i> Cancel
        </button>
        <button class="btn-save" data-action="save-resend">
            <i class="fas fa-paper-plane"></i> Save & Resend
        </button>
    `;

    editContainer.appendChild(textarea);
    editContainer.appendChild(editActions);

    // Hide original content and footer
    msgText.style.display = 'none';
    if (msgFooter) msgFooter.style.display = 'none';

    // Add edit UI
    msgElement.querySelector('.message-content').appendChild(editContainer);

    // Focus and select
    textarea.focus();
    textarea.setSelectionRange(textarea.value.length, textarea.value.length);

    // Auto-resize textarea
    textarea.addEventListener('input', () => {
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 300) + 'px';
    });
}

async function saveAndResend(msgElement, newText) {
    if (!newText || !newText.trim()) {
        showToast('Message cannot be empty', 'error');
        return;
    }

    if (isStreaming) {
        showToast('Cannot resend while streaming', 'error');
        return;
    }

    const chatMessages = document.getElementById('chat-messages');
    if (!chatMessages) return;

    const allMessages = Array.from(chatMessages.querySelectorAll('.message'));
    const msgIndex = allMessages.indexOf(msgElement);

    // Remove all messages after (and including) this one
    for (let i = msgIndex; i < allMessages.length; i++) {
        allMessages[i].remove();
    }

    // Send new message
    const input = document.getElementById('user-input');
    if (input) {
        input.value = newText.trim();
        updateSendButton();
        await sendMessage();
    }

    showToast('Message edited and resent', 'success');
}

function cancelEdit(msgElement) {
    const editContainer = msgElement.querySelector('.edit-message-container');
    const msgText = msgElement.querySelector('.message-text');
    const msgFooter = msgElement.querySelector('.message-footer');

    if (editContainer) editContainer.remove();
    if (msgText) msgText.style.display = '';
    if (msgFooter) msgFooter.style.display = '';
}

function renderGraphVisualization(container, graphData) {
    if (!container || typeof vis === 'undefined') {
        if (container) {
            container.innerHTML = '<p style="text-align: center; color: #ef4444; padding: 2rem;">Graph library not loaded</p>';
        }
        return;
    }

    try {
        const nodes = new vis.DataSet(graphData.nodes.map(node => ({
            id: node.id,
            label: node.label,
            title: node.title,
            color: node.type === 'main' ? '#ef4444' : '#f59e0b',
            size: node.type === 'main' ? 30 : 20,
            font: { size: 12, color: '#1f2937' },
            fixed: { x: false, y: false }
        })));

        const edges = new vis.DataSet(graphData.edges.map(edge => ({
            from: edge.from,
            to: edge.to,
            arrows: 'to',
            color: edge.type === 'hierarchical' ? '#3b82f6' : edge.type === 'semantic' ? '#10b981' : '#a855f7',
            smooth: { type: 'continuous' }
        })));

        const network = new vis.Network(container, { nodes, edges }, {
            physics: {
                enabled: true,
                stabilization: {
                    enabled: true,
                    iterations: 200,
                    fit: true
                },
                solver: 'forceAtlas2Based',
                forceAtlas2Based: {
                    gravitationalConstant: -50,
                    centralGravity: 0.01,
                    springLength: 100,
                    springConstant: 0.08
                }
            },
            interaction: {
                hover: true,
                tooltipDelay: 100,
                dragNodes: false,
                dragView: true,
                zoomView: true
            }
        });

        // Stop physics after stabilization
        network.on('stabilizationIterationsDone', function () {
            network.setOptions({ physics: false });
        });

        // Fallback: stop after 3 seconds
        setTimeout(() => {
            network.setOptions({ physics: false });
        }, 3000);

    } catch (error) {
        console.error('Error rendering graph:', error);
        container.innerHTML = '<p style="text-align: center; color: #ef4444; padding: 2rem;">Error rendering graph</p>';
    }
}

function displayMessage(msg) {
    const chatMessages = document.getElementById('chat-messages');
    if (!chatMessages) return;

    const msgEl = createMessageElement(msg.role, msg.content, msg.timestamp);

    // Display attachments from session storage
    if (msg.attachments && msg.attachments.length > 0) {
        const attachmentsDiv = document.createElement('div');
        attachmentsDiv.className = 'message-attachments';

        msg.attachments.forEach(att => {
            const attachment = document.createElement('div');
            const isImage = att.filename.match(/\.(jpg|jpeg|png|gif|webp)$/i);

            if (isImage) {
                attachment.className = 'attachment-image';
                attachment.innerHTML = `<img src="/api/attachment/${att.path}" alt="${escapeHtml(att.filename)}">`;
            } else {
                attachment.className = 'attachment-doc';
                attachment.innerHTML = `
                    <i class="fas fa-file-alt"></i>
                    <span>${escapeHtml(att.filename)}</span>
                `;
            }

            attachmentsDiv.appendChild(attachment);
        });

        const msgContent = msgEl.querySelector('.message-content');
        msgContent.insertBefore(attachmentsDiv, msgContent.firstChild);
    }
    // Fallback to old images field for backward compatibility
    else if (msg.role === 'user' && msg.images && msg.images.length > 0) {
        const attachmentsDiv = document.createElement('div');
        attachmentsDiv.className = 'message-attachments';

        msg.images.forEach(filename => {
            const attachment = document.createElement('div');
            const isImage = filename.match(/\.(jpg|jpeg|png|gif|webp)$/i);

            if (isImage) {
                attachment.className = 'attachment-image';
                attachment.innerHTML = `<img src="/uploads/${filename}" alt="${escapeHtml(filename)}">`;
            } else {
                attachment.className = 'attachment-doc';
                attachment.innerHTML = `
                    <i class="fas fa-file-alt"></i>
                    <span>${escapeHtml(filename)}</span>
                `;
            }

            attachmentsDiv.appendChild(attachment);
        });

        const msgContent = msgEl.querySelector('.message-content');
        msgContent.insertBefore(attachmentsDiv, msgContent.firstChild);
    }

    if (msg.role === 'assistant') {
        const msgText = msgEl.querySelector('.message-text');
        msgText.innerHTML = formatMarkdown(msg.content);
        renderEnhancements(msgText); // Render mermaid diagrams, math, etc.

        if (msg.graph_data && msg.graph_data.nodes && msg.graph_data.nodes.length > 0) {
            const stats = {
                nodes: msg.graph_data.nodes.length,
                connected: msg.graph_data.nodes.filter(n => n.type === 'connected').length,
                images: msg.images ? msg.images.length : 0
            };
            displayGraph(msg.graph_data, stats);
        }

        if (msg.images && msg.images.length > 0) {
            displayImages(msg.images);
        }
    }

    chatMessages.appendChild(msgEl);
}

function displayGraph(graphData, stats) {
    const chatMessages = document.getElementById('chat-messages');
    if (!chatMessages) return;

    // Store graph reference to prevent removal
    const graphDiv = document.createElement('div');
    graphDiv.className = 'graph-visualization';
    graphDiv.setAttribute('data-persistent', 'true');
    const graphId = 'graph-' + Date.now();
    graphDiv.innerHTML = `
        <div class="graph-header">
            <h3><i class="fas fa-project-diagram"></i> Knowledge Graph</h3>
            <div class="graph-stats">
                <span><i class="fas fa-circle-nodes"></i> ${stats.nodes} nodes</span>
                <span><i class="fas fa-link"></i> ${stats.connected} connected</span>
                <span><i class="fas fa-image"></i> ${stats.images} images</span>
            </div>
        </div>
        <div class="graph-network" id="${graphId}"></div>
        <div class="graph-legend">
            <div class="legend-item"><span class="legend-dot main"></span> Main Node</div>
            <div class="legend-item"><span class="legend-dot connected"></span> Connected Nodes</div>
        </div>
    `;

    chatMessages.appendChild(graphDiv);

    // Render graph using the function
    setTimeout(() => {
        const container = document.getElementById(graphId);
        if (container) {
            renderGraphVisualization(container, graphData);
        }
    }, 100);
}

function displayImages(imagePaths) {
    const chatMessages = document.getElementById('chat-messages');
    if (!chatMessages) return;

    const galleryDiv = document.createElement('div');
    galleryDiv.className = 'image-gallery';
    galleryDiv.innerHTML = `
        <div class="gallery-header">
            <i class="fas fa-images"></i> Retrieved Images (${imagePaths.length})
        </div>
        <div class="gallery-grid">
            ${imagePaths.map(path => `
                <div class="gallery-item">
                    <img src="/api/image?path=${encodeURIComponent(path)}" alt="Image" loading="lazy">
                </div>
            `).join('')}
        </div>
    `;

    chatMessages.appendChild(galleryDiv);
}

// Display images within a specific message (for knowledge graph images)
function displayImagesInMessage(messageContent, imagePaths) {
    if (!messageContent || !imagePaths || imagePaths.length === 0) return;

    const galleryDiv = document.createElement('div');
    galleryDiv.className = 'image-gallery';
    galleryDiv.innerHTML = `
        <div class="gallery-header">
            <i class="fas fa-images"></i> Related Figures (${imagePaths.length})
        </div>
        <div class="gallery-grid">
            ${imagePaths.map(path => `
                <div class="gallery-item">
                    <img src="/api/image?path=${encodeURIComponent(path)}"
                         alt="Figure from knowledge graph"
                         loading="lazy"
                         onerror="this.parentElement.style.display='none'">
                </div>
            `).join('')}
        </div>
    `;

    // Insert after the message text but before action buttons
    const actionButtons = messageContent.querySelector('.message-actions');
    if (actionButtons) {
        messageContent.insertBefore(galleryDiv, actionButtons);
    } else {
        messageContent.appendChild(galleryDiv);
    }
}

async function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`/api/upload/${userId}`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        uploadedFiles.push(data);

        const preview = document.getElementById('file-preview');
        if (preview) {
            const isImage = file.type.startsWith('image/');
            const chip = document.createElement('div');
            chip.className = 'file-attachment';

            if (isImage) {
                // Create image preview
                const reader = new FileReader();
                reader.onload = (e) => {
                    chip.innerHTML = `
                        <img src="${e.target.result}" alt="${escapeHtml(file.name)}">
                        <button class="remove-attachment" onclick="removeFile('${data.file_id}')">
                            <i class="fas fa-times"></i>
                        </button>
                    `;
                };
                reader.readAsDataURL(file);
            } else {
                // Create document icon
                chip.innerHTML = `
                    <div class="doc-attachment">
                        <i class="fas fa-file-alt"></i>
                        <span class="doc-name">${escapeHtml(file.name)}</span>
                    </div>
                    <button class="remove-attachment" onclick="removeFile('${data.file_id}')">
                        <i class="fas fa-times"></i>
                    </button>
                `;
            }

            preview.appendChild(chip);
        }

        updateSendButton();
        e.target.value = ''; // Clear file input
    } catch (error) {
        console.error('Error uploading file:', error);
    }
}

function removeFile(fileId) {
    uploadedFiles = uploadedFiles.filter(f => f.file_id !== fileId);
    const preview = document.getElementById('file-preview');
    if (preview) {
        Array.from(preview.children).forEach(chip => {
            if (chip.innerHTML.includes(fileId)) chip.remove();
        });
    }
    updateSendButton();
}

async function showProfile() {
    try {
        const response = await fetch(`/api/profile/${userId}`);
        const data = await response.json();

        const username = document.getElementById('profile-username');
        if (username) username.value = data.username;

        const email = document.getElementById('profile-email');
        if (email) email.value = data.email;

        const fullname = document.getElementById('profile-fullname');
        if (fullname) fullname.value = data.full_name || '';

        const modal = document.getElementById('profile-modal');
        if (modal) modal.classList.add('active');
    } catch (error) {
        console.error('Error loading profile:', error);
    }
}

async function saveProfile() {
    const emailEl = document.getElementById('profile-email');
    const fullNameEl = document.getElementById('profile-fullname');
    const email = emailEl ? emailEl.value : '';
    const fullName = fullNameEl ? fullNameEl.value : '';

    try {
        await fetch(`/api/profile/${userId}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, full_name: fullName })
        });

        alert('Profile updated');
    } catch (error) {
        console.error('Error saving profile:', error);
    }
}

function logout() {
    localStorage.clear();
    window.location.href = '/';
}

async function showFAQs() {
    try {
        const response = await fetch('/api/faqs');
        const data = await response.json();

        const content = document.getElementById('faq-content');
        if (content) {
            content.innerHTML = data.faqs.map(faq => `
                <div class="faq-item" onclick="this.classList.toggle('active')">
                    <div class="faq-question">
                        ${escapeHtml(faq.question)}
                        <i class="fas fa-chevron-down"></i>
                    </div>
                    <div class="faq-answer">${escapeHtml(faq.answer)}</div>
                </div>
            `).join('');
        }

        const modal = document.getElementById('faq-modal');
        if (modal) modal.classList.add('active');
    } catch (error) {
        console.error('Error loading FAQs:', error);
    }
}

// Initialize Mermaid
if (typeof mermaid !== 'undefined') {
    mermaid.initialize({
        startOnLoad: false,
        theme: 'default',
        securityLevel: 'loose',
        flowchart: {
            useMaxWidth: true,
            htmlLabels: true
        }
    });
}

// Configure Marked.js with custom renderer
if (typeof marked !== 'undefined') {
    const renderer = new marked.Renderer();

    // Custom code block renderer to handle mermaid diagrams
    renderer.code = function(code, language) {
        if (language === 'mermaid') {
            const mermaidId = 'mermaid-' + Math.random().toString(36).substr(2, 9);
            // Return mermaid div that will be rendered later
            return `<div class="mermaid" id="${mermaidId}">${code}</div>`;
        } else {
            // Regular code block with syntax highlighting
            const codeId = 'code-' + Math.random().toString(36).substr(2, 9);
            const displayLang = language || 'plaintext';
            return `<div class="code-block-wrapper">
                <div class="code-block-header">
                    <span class="code-language">${displayLang}</span>
                    <button class="copy-code-btn" data-action="copy-code" data-code-id="${codeId}">
                        <i class="fas fa-copy"></i> Copy
                    </button>
                </div>
                <pre><code id="${codeId}" class="language-${language || 'plaintext'}">${code}</code></pre>
            </div>`;
        }
    };

    // Custom image renderer to handle both external and local images
    renderer.image = function(href, title, text) {
        return `<img src="${href}" alt="${text || ''}" title="${title || ''}" style="max-width: 100%; height: auto; border-radius: 8px; margin: 10px 0;" />`;
    };

    // Custom link renderer
    renderer.link = function(href, title, text) {
        return `<a href="${href}" title="${title || ''}" target="_blank" rel="noopener noreferrer">${text}</a>`;
    };

    marked.setOptions({
        renderer: renderer,
        highlight: function(code, lang) {
            if (typeof hljs !== 'undefined' && lang && hljs.getLanguage(lang)) {
                try {
                    return hljs.highlight(code, { language: lang }).value;
                } catch (err) {
                    console.error('Highlight.js error:', err);
                }
            }
            return code;
        },
        breaks: true,
        gfm: true,
        headerIds: true,
        mangle: false
    });
}

function formatMarkdown(text) {
    if (!text) return '';

    // Use marked.js if available, otherwise fallback to basic rendering
    if (typeof marked !== 'undefined' && typeof DOMPurify !== 'undefined') {
        try {
            // Parse markdown with marked.js
            let html = marked.parse(text);

            // Sanitize HTML with DOMPurify
            html = DOMPurify.sanitize(html, {
                ADD_TAGS: ['iframe'],
                ADD_ATTR: ['allow', 'allowfullscreen', 'frameborder', 'scrolling']
            });

            return html;
        } catch (err) {
            console.error('Marked.js error:', err);
            return formatMarkdownBasic(text); // Fallback to basic
        }
    } else {
        return formatMarkdownBasic(text); // Fallback to basic
    }
}

// Fallback basic markdown formatter (original implementation)
function formatMarkdownBasic(text) {
    let html = escapeHtml(text);

    html = html.replace(/^#### (.+)$/gm, '<h4>$1</h4>');
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

    // Code blocks with language detection and copy button
    html = html.replace(/```(\w+)?\n?([\s\S]*?)```/g, (match, lang, code) => {
        const language = lang || 'plaintext';
        const codeId = 'code-' + Math.random().toString(36).substr(2, 9);
        return `<div class="code-block-wrapper"><div class="code-block-header"><span class="code-language">${language}</span><button class="copy-code-btn" data-action="copy-code" data-code-id="${codeId}"><i class="fas fa-copy"></i> Copy</button></div><pre><code id="${codeId}" class="language-${language}">${code.trim()}</code></pre></div>`;
    });

    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');

    const lines = html.split('\n');
    let inTable = false;
    let tableRows = [];

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();

        if (line.startsWith('|') && line.endsWith('|')) {
            const cells = line.split('|').filter(c => c.trim()).map(c => c.trim());
            if (cells.every(c => /^[-:]+$/.test(c))) continue;

            if (!inTable) {
                inTable = true;
                tableRows = [];
            }
            tableRows.push(cells);
        } else if (inTable) {
            if (tableRows.length > 0) {
                let table = '<table>';
                tableRows.forEach((row, idx) => {
                    table += '<tr>';
                    const tag = idx === 0 ? 'th' : 'td';
                    row.forEach(cell => {
                        table += `<${tag}>${cell}</${tag}>`;
                    });
                    table += '</tr>';
                });
                table += '</table>';

                lines[i - tableRows.length - 1] = table;
                for (let j = i - tableRows.length; j < i; j++) {
                    lines[j] = '';
                }
            }
            inTable = false;
            tableRows = [];
        }
    }

    html = lines.join('\n');
    html = html.replace(/^\d+\.\s+(.+)$/gm, '<li>$1</li>');
    html = html.replace(/^[-*]\s+(.+)$/gm, '<li>$1</li>');
    html = html.replace(/(<li>.*?<\/li>\n?)+/g, match => `<ul>${match}</ul>`);

    html = html.replace(/\n\n/g, '</p><p>');
    html = '<p>' + html + '</p>';
    html = html.replace(/<p><\/p>/g, '');
    html = html.replace(/<p>(<h[1-4]>)/g, '$1');
    html = html.replace(/(<\/h[1-4]>)<\/p>/g, '$1');
    html = html.replace(/<p>(<table>)/g, '$1');
    html = html.replace(/(<\/table>)<\/p>/g, '$1');
    html = html.replace(/<p>(<ul>)/g, '$1');
    html = html.replace(/(<\/ul>)<\/p>/g, '$1');

    return html;
}

// Render Mermaid diagrams and KaTeX math in a message element
function renderEnhancements(element, skipMermaid = false) {
    // Render Mermaid diagrams - SKIP during streaming to prevent flickering
    if (typeof mermaid !== 'undefined' && !skipMermaid) {
        const mermaidElements = element.querySelectorAll('.mermaid');
        mermaidElements.forEach((el, index) => {
            try {
                // Only render if not already rendered
                if (!el.getAttribute('data-processed')) {
                    mermaid.run({
                        nodes: [el]
                    }).then(() => {
                        el.setAttribute('data-processed', 'true');
                    }).catch(err => {
                        console.error('Mermaid rendering error:', err);
                        el.innerHTML = `<pre style="color: red;">Mermaid diagram error: ${err.message}</pre>`;
                    });
                }
            } catch (err) {
                console.error('Mermaid error:', err);
            }
        });
    }

    // Render KaTeX math
    if (typeof renderMathInElement !== 'undefined') {
        try {
            renderMathInElement(element, {
                delimiters: [
                    {left: '$$', right: '$$', display: true},
                    {left: '$', right: '$', display: false},
                    {left: '\\(', right: '\\)', display: false},
                    {left: '\\[', right: '\\]', display: true}
                ],
                throwOnError: false
            });
        } catch (err) {
            console.error('KaTeX rendering error:', err);
        }
    }

    // Highlight code blocks
    if (typeof hljs !== 'undefined') {
        element.querySelectorAll('pre code').forEach((block) => {
            if (!block.getAttribute('data-highlighted')) {
                hljs.highlightElement(block);
                block.setAttribute('data-highlighted', 'true');
            }
        });
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatTime(isoString) {
    const date = new Date(isoString);
    const now = new Date();
    const diff = now - date;
    const mins = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);

    if (mins < 1) return 'Just now';
    if (mins < 60) return `${mins}m ago`;
    if (hours < 24) return `${hours}h ago`;
    if (days < 7) return `${days}d ago`;

    return date.toLocaleDateString();
}

// =========================================================
// Test Plan Export Functions
// =========================================================

// Track the last test plan query for export
let lastTestPlanQuery = null;

function isTestPlanContent(content) {
    // Check if the content looks like a test plan response
    const testPlanIndicators = [
        'test plan',
        'test case',
        'Test Case',
        'TEST PLAN',
        'requirement',
        'procedure',
        'pass criteria',
        'EMC test'
    ];

    const lowerContent = content.toLowerCase();
    let matches = 0;
    for (const indicator of testPlanIndicators) {
        if (lowerContent.includes(indicator.toLowerCase())) {
            matches++;
        }
    }
    return matches >= 3;
}

function addExportButton(messageElement, query, downloadAvailable = false, downloadUrl = null, fileName = null, messageId = null) {
    const msgContent = messageElement.querySelector('.message-content');
    if (!msgContent) return;

    // Check if export button already exists
    if (msgContent.querySelector('.export-actions')) return;

    const escapedQuery = escapeHtml(query.replace(/'/g, "\\'"));

    const exportDiv = document.createElement('div');
    exportDiv.className = 'export-actions';

    // ChatGPT-style download button (primary) if available
    let buttons = '';
    if (downloadAvailable && downloadUrl) {
        buttons += `
            <button class="export-btn download-btn-primary" onclick="downloadGeneratedTestPlan('${downloadUrl}', '${fileName || 'test_plan.docx'}')">
                <i class="fas fa-download"></i> Download Test Plan
            </button>
        `;
    }

    // Alternative: Generate new report button (secondary)
    buttons += `
        <button class="export-btn docx-btn-secondary" onclick="downloadTestPlanReport()">
            <i class="fas fa-file-word"></i> Generate New Report
        </button>
    `;

    exportDiv.innerHTML = `
        <div class="export-toolbar">
            ${buttons}
        </div>
    `;

    msgContent.appendChild(exportDiv);
}

// function exportTestPlanPDF removed

// ChatGPT-style download function
async function downloadGeneratedTestPlan(downloadUrl, fileName) {
    try {
        // Add session_id to URL if we have one
        const url = currentSessionId ? `${downloadUrl}?session_id=${currentSessionId}` : downloadUrl;

        // Trigger download
        const link = document.createElement('a');
        link.href = url;
        link.download = fileName;
        link.click();

        console.log('Download initiated:', fileName);
    } catch (error) {
        console.error('Download failed:', error);
        alert('Failed to download test plan. Please try again.');
    }
}

async function downloadTestPlanReport() {
    const btn = event.target.closest('.export-btn');
    const originalHtml = btn.innerHTML;

    // Find the message content
    const msgContent = btn.closest('.message-content');
    const msgText = msgContent.querySelector('.message-text');
    const content = msgText.dataset.rawText || msgText.innerText;

    // Extract title from content (first line usually)
    let title = "EMC_Test_Plan";
    const lines = content.split('\n');
    for (const line of lines) {
        if (line.trim().startsWith('# ')) {
            title = line.trim().substring(2).trim();
            break;
        }
    }

    try {
        // Show loading state
        btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating...';
        btn.disabled = true;

        const response = await fetch('/api/test-plan/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                title: title,
                content: content
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to generate report');
        }

        // Get the blob
        const blob = await response.blob();

        // Get filename
        const contentDisposition = response.headers.get('Content-Disposition');
        let filename = `${title.replace(/[^a-zA-Z0-9]/g, '_')}.docx`;
        if (contentDisposition) {
            const match = contentDisposition.match(/filename=(.+)/);
            if (match) filename = match[1];
        }

        // Create download link
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);

        // Show success
        btn.innerHTML = '<i class="fas fa-check"></i> Downloaded!';
        setTimeout(() => {
            btn.innerHTML = originalHtml;
            btn.disabled = false;
        }, 2000);

    } catch (error) {
        console.error('Report generation failed:', error);
        btn.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Failed';
        setTimeout(() => {
            btn.innerHTML = originalHtml;
            btn.disabled = false;
        }, 2000);
        alert('Failed to generate report: ' + error.message);
    }
}


