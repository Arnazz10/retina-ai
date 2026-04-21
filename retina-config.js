window.RETINA_CONFIG = {
  frontendOnly: true,
  apiBase: window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' 
    ? "http://127.0.0.1:5000" 
    : "" // On production, assume same origin or configured elsewhere
};
