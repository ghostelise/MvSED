const navToggle = document.querySelector('.nav-toggle');
const navLinks = document.querySelector('.nav-links');

if (navToggle && navLinks) {
  navToggle.addEventListener('click', () => {
    const isOpen = navLinks.classList.toggle('open');
    navToggle.setAttribute('aria-expanded', String(isOpen));
  });

  navLinks.querySelectorAll('a').forEach((link) => {
    link.addEventListener('click', () => {
      navLinks.classList.remove('open');
      navToggle.setAttribute('aria-expanded', 'false');
    });
  });
}

const copyButton = document.querySelector('[data-copy-bibtex]');
const bibtex = document.getElementById('bibtex');

if (copyButton && bibtex) {
  copyButton.addEventListener('click', async () => {
    const originalLabel = copyButton.textContent;
    const text = bibtex.textContent.trim();

    try {
      await navigator.clipboard.writeText(text);
    } catch (error) {
      const helper = document.createElement('textarea');
      helper.value = text;
      helper.setAttribute('readonly', '');
      helper.style.position = 'fixed';
      helper.style.opacity = '0';
      document.body.appendChild(helper);
      helper.select();
      document.execCommand('copy');
      helper.remove();
    }

    copyButton.textContent = 'Copied';
    window.setTimeout(() => {
      copyButton.textContent = originalLabel;
    }, 1800);
  });
}

const repositoryLinks = document.querySelectorAll('[data-repository-link]');

if (repositoryLinks.length && window.location.hostname.endsWith('.github.io')) {
  const owner = window.location.hostname.split('.')[0];
  const pathParts = window.location.pathname.split('/').filter(Boolean);
  const repository = pathParts[0] || `${owner}.github.io`;
  const codeUrl = `https://github.com/${owner}/${repository}/tree/main/code`;

  repositoryLinks.forEach((link) => {
    link.href = codeUrl;
  });
}
