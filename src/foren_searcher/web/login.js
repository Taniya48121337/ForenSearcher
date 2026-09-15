'use strict';
document.querySelector('#login-form').addEventListener('submit', async event => {
  event.preventDefault();
  const button = event.target.querySelector('button');
  const error = document.querySelector('#login-error');
  button.disabled = true; error.textContent = '';
  try {
    const response = await fetch('/api/session', {
      method: 'POST', headers: {'Content-Type': 'application/json', 'X-Foren-Request': '1'},
      body: JSON.stringify({code: document.querySelector('#invite-code').value.trim()})
    });
    if (!response.ok) { const data = await response.json(); throw new Error(data.detail || '暂时无法登录'); }
    location.reload();
  } catch (exception) { error.textContent = exception.message || '连接中断，请重试'; }
  finally { button.disabled = false; }
});
