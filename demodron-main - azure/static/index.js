(function() {
  'use strict';
  
  // Security check
  const isSecure = window.isSecureContext || location.protocol === 'https:' || location.hostname === 'localhost';
  if (!isSecure) document.getElementById('securityWarning').style.display = 'block';
  
  // Device ID
  function generateDeviceId() {
    let id = localStorage.getItem('device_id');
    if (!id) {
      id = 'dev_' + Math.random().toString(36).substr(2, 9);
      localStorage.setItem('device_id', id);
    }
    return id;
  }
  const deviceId = generateDeviceId();
  
  // DOM
  const video = document.getElementById('video');
  const overlay = document.getElementById('overlay');
  const octx = overlay.getContext('2d');
  const statusBadge = document.getElementById('statusBadge');
  const deviceSelect = document.getElementById('deviceSelect');
  
  // State
  let socket;
  let stream;
  let capturing = false;
  let sending = false;
  let animationId;
  
  // No special detection tracking needed
  
  // Performance state (auto-tuned)
  let sendWidth = 960;
  let maxCadenceHz = 20;
  let jpegQuality = 0.85;  // Increased from 0.70 for better quality
  let lastCaptureTime = 0;
  let framesSent = 0;
  let framesDropped = 0;
  let lastFpsTime = Date.now();
  let lastFpsCount = 0;
  let lastBlobSize = 0;
  let lastInferMs = 0;
  
  // Geolocation state
  let gpsLat, gpsLon, gpsAcc, gpsWatchId;
  let heading, pitch, roll;
  let hfov = 63;
  
  // Map
  let map, observerMarker, bearingLayer;
  
  // Socket connection
  function connectSocket() {
    socket = io({ transports: ['websocket'], reconnection: true, reconnectionDelay: 1000 });
    
    socket.on('connect', () => {
      console.log('[SOCKET] Connected');
      statusBadge.textContent = 'Connected';
      statusBadge.className = 'status-badge status-connected';
    });
    
    socket.on('disconnect', () => {
      console.log('[SOCKET] Disconnected');
      statusBadge.textContent = 'Disconnected';
      statusBadge.className = 'status-badge status-disconnected';
    });
    
    socket.on('predictions', (data) => {
      drawPredictions(data);
      if (data.infer_ms !== undefined) {
        lastInferMs = data.infer_ms;
        document.getElementById('perfInfer').textContent = data.infer_ms + 'ms';
      }
      
      // Send observation if we have GPS and boxes
      if (gpsLat && gpsLon && data.boxes && data.boxes.length > 0) {
        sendObservations(data.boxes);
      }
    });
    
    socket.on('perf_advice', (advice) => {
      console.log('[PERF] Advice:', advice);
      if (advice.sendWidth) sendWidth = advice.sendWidth;
      if (advice.maxCadenceHz) maxCadenceHz = advice.maxCadenceHz;
      if (advice.jpegQuality) jpegQuality = advice.jpegQuality;
    });
    
    socket.on('detections_update', () => loadDetections());
    socket.on('target_update', (target) => displayUAVTarget(target));
  }
  
  // Camera
  async function enumerateDevices() {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(d => d.kind === 'videoinput');
      deviceSelect.innerHTML = '<option value="">Default Camera</option>';
      
      let rearCameraId = null;
      
      videoDevices.forEach((device, idx) => {
        const option = document.createElement('option');
        option.value = device.deviceId;
        option.textContent = device.label || `Camera ${idx + 1}`;
        deviceSelect.appendChild(option);
        
        // Try to detect rear camera
        const label = (device.label || '').toLowerCase();
        if (label.includes('back') || label.includes('rear') || label.includes('environment')) {
          rearCameraId = device.deviceId;
        }
      });
      
      // Auto-select rear camera if found
      if (rearCameraId) {
        deviceSelect.value = rearCameraId;
        console.log('[CAMERA] Auto-selected rear camera');
      }
    } catch (err) {
      console.error('[CAMERA] Enumeration error:', err);
    }
  }
  
  async function startCamera(constraints) {
    try {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
      }
      video.srcObject = null;
      await new Promise(resolve => setTimeout(resolve, 100));
      
      stream = await navigator.mediaDevices.getUserMedia(constraints);
      video.srcObject = stream;
      
      await new Promise((resolve, reject) => {
        video.onloadedmetadata = resolve;
        video.onerror = reject;
        setTimeout(() => reject(new Error('Timeout')), 5000);
      });
      
      await video.play();
      const vw = video.videoWidth || 640;
      const vh = video.videoHeight || 480;
      overlay.width = vw;
      overlay.height = vh;
      
      // Enable tap-to-focus
      setupTapToFocus();
      
      console.log(`[CAMERA] Stream started: ${vw}x${vh}`);
      await enumerateDevices();
      return true;
    } catch (err) {
      console.error('[CAMERA] Error:', err);
      alert(`Camera Error: ${err.message}`);
      return false;
    }
  }
  
  // Capture loop using requestVideoFrameCallback (rVFC)
  function captureLoop(now, metadata) {
    if (!capturing) return;
    
    // Check cadence (auto-tuned by server)
    const elapsed = (now - lastCaptureTime) / 1000;
    const minInterval = 1.0 / maxCadenceHz;
    
    if (elapsed < minInterval) {
      video.requestVideoFrameCallback(captureLoop);
      return;
    }
    
    lastCaptureTime = now;
    
    // Skip if already sending (queue-latest on client side too)
    if (sending) {
      framesDropped++;
      video.requestVideoFrameCallback(captureLoop);
      return;
    }
    
    // Capture and send
    captureAndSend();
    
    // Continue loop
    video.requestVideoFrameCallback(captureLoop);
  }
  
  async function captureAndSend() {
    if (!stream || !video.videoWidth || sending) return;
    
    sending = true;
    
    try {
      const vw = video.videoWidth;
      const vh = video.videoHeight;
      const targetHeight = Math.round(sendWidth * vh / vw);
      
      // Use OffscreenCanvas for better performance
      const canvas = new OffscreenCanvas(sendWidth, targetHeight);
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, sendWidth, targetHeight);
      
      // Encode to JPEG (auto-tuned quality)
      const blob = await canvas.convertToBlob({ type: 'image/jpeg', quality: jpegQuality });
      lastBlobSize = blob.size;
      
      // Convert to ArrayBuffer (NOT base64 - lower latency)
      const arrayBuffer = await blob.arrayBuffer();
      
      // Send via Socket.IO
      socket.emit('frame', { w: sendWidth, h: targetHeight, jpg: arrayBuffer });
      framesSent++;
      
      // Update performance stats
      const now = Date.now();
      if (now - lastFpsTime >= 1000) {
        const actualFps = (framesSent - lastFpsCount) / ((now - lastFpsTime) / 1000);
        const dropRate = framesDropped / (framesSent + framesDropped);
        
        document.getElementById('perfFps').textContent = actualFps.toFixed(1);
        document.getElementById('perfDropped').textContent = (dropRate * 100).toFixed(0) + '%';
        document.getElementById('perfSize').textContent = (lastBlobSize / 1024).toFixed(1) + 'KB';
        
        lastFpsTime = now;
        lastFpsCount = framesSent;
      }
    } catch (err) {
      console.error('[CAPTURE] Error:', err);
    } finally {
      sending = false;
    }
  }
  
  function startCapture() {
    if (!stream || !socket || !socket.connected) {
      alert('Camera not ready or socket not connected');
      return;
    }
    
    capturing = true;
    framesSent = 0;
    framesDropped = 0;
    lastFpsTime = Date.now();
    lastFpsCount = 0;
    lastCaptureTime = performance.now();
    
    // Use rVFC if available (Chrome/Edge), otherwise fallback
    if ('requestVideoFrameCallback' in video) {
      video.requestVideoFrameCallback(captureLoop);
      console.log('[CAPTURE] Started with rVFC');
    } else {
      animationId = setInterval(captureAndSend, 1000 / maxCadenceHz);
      console.log('[CAPTURE] Started with setInterval (rVFC not supported)');
    }
  }
  
  function stopCapture() {
    capturing = false;
    if (animationId) {
      clearInterval(animationId);
      animationId = null;
    }
    console.log('[CAPTURE] Stopped');
  }
  
  // Tap-to-focus functionality
  function setupTapToFocus() {
    if (!stream) return;
    
    const videoTrack = stream.getVideoTracks()[0];
    if (!videoTrack) return;
    
    // Check if focus mode is supported
    const capabilities = videoTrack.getCapabilities();
    if (!capabilities || !capabilities.focusMode) {
      console.log('[FOCUS] Tap-to-focus not supported on this device');
      return;
    }
    
    // Handler for tap/click events
    const handleTap = async (e) => {
      e.preventDefault();
      
      // Get tap coordinates relative to video element
      const rect = video.getBoundingClientRect();
      const x = (e.clientX || e.touches?.[0]?.clientX || 0) - rect.left;
      const y = (e.clientY || e.touches?.[0]?.clientY || 0) - rect.top;
      
      // Normalize to 0-1 range
      const focusX = Math.max(0, Math.min(1, x / rect.width));
      const focusY = Math.max(0, Math.min(1, y / rect.height));
      
      try {
        // Apply focus at tap point
        const constraints = {
          advanced: [{
            focusMode: 'manual',
            focusDistance: 0, // Will be ignored, but required for some browsers
          }]
        };
        
        // Try point focus if available
        if (capabilities.focusMode.includes('single-shot')) {
          constraints.advanced[0].focusMode = 'single-shot';
        }
        
        await videoTrack.applyConstraints(constraints);
        
        // Show focus indicator
        showFocusIndicator(x + rect.left, y + rect.top);
        
        console.log(`[FOCUS] Focused at (${(focusX * 100).toFixed(0)}%, ${(focusY * 100).toFixed(0)}%)`);
      } catch (err) {
        console.error('[FOCUS] Error applying focus:', err);
      }
    };
    
    // Add event listeners (both touch and mouse)
    video.addEventListener('click', handleTap);
    video.addEventListener('touchstart', handleTap);
    
    console.log('[FOCUS] Tap-to-focus enabled');
  }
  
  // Show visual feedback for focus point
  function showFocusIndicator(x, y) {
    // Remove existing indicator
    const existing = document.getElementById('focusIndicator');
    if (existing) existing.remove();
    
    // Create focus indicator
    const indicator = document.createElement('div');
    indicator.id = 'focusIndicator';
    indicator.style.position = 'fixed';
    indicator.style.left = `${x}px`;
    indicator.style.top = `${y}px`;
    indicator.style.width = '60px';
    indicator.style.height = '60px';
    indicator.style.border = '2px solid #09ECFD';
    indicator.style.borderRadius = '50%';
    indicator.style.transform = 'translate(-50%, -50%)';
    indicator.style.pointerEvents = 'none';
    indicator.style.zIndex = '9999';
    indicator.style.transition = 'opacity 0.5s, transform 0.3s';
    indicator.style.boxShadow = '0 0 10px rgba(9, 236, 253, 0.5)';
    
    document.body.appendChild(indicator);
    
    // Animate
    requestAnimationFrame(() => {
      indicator.style.transform = 'translate(-50%, -50%) scale(0.8)';
    });
    
    // Remove after animation
    setTimeout(() => {
      indicator.style.opacity = '0';
      setTimeout(() => indicator.remove(), 500);
    }, 800);
  }
  
  // Drawing predictions (normalized 0-1 coords)
  function drawPredictions(data) {
    octx.clearRect(0, 0, overlay.width, overlay.height);
    
    if (data.error) {
      console.error('[PREDICTIONS] Error:', data.error);
      document.getElementById('detCount').textContent = '0';
      return;
    }
    
    const boxes = data.boxes || [];
    document.getElementById('detCount').textContent = boxes.length;
    
    if (boxes.length === 0) return;
    
    const W = overlay.width;
    const H = overlay.height;
    
    octx.font = 'bold 14px -apple-system, sans-serif';
    octx.lineWidth = 3;
    
    boxes.forEach((box, index) => {
      const x1 = Math.round(box.x1n * W);
      const y1 = Math.round(box.y1n * H);
      const x2 = Math.round(box.x2n * W);
      const y2 = Math.round(box.y2n * H);
      const w = x2 - x1;
      const h = y2 - y1;
      
      if (w < 2 || h < 2) return;
      
      // Red box
      octx.strokeStyle = '#FF0000';
      octx.strokeRect(x1, y1, w, h);
      
      // Label with index number
      const droneIndex = index + 1;
      const label = `Drone #${droneIndex} ${box.conf.toFixed(2)}`;
      const pad = 6;
      const th = 22;
      const tw = octx.measureText(label).width + 2 * pad;
      
      // Red background for label
      octx.fillStyle = '#DC0000';
      octx.fillRect(x1, Math.max(0, y1 - th), tw, th);
      
      octx.fillStyle = '#fff';
      octx.fillText(label, x1 + pad, Math.max(14, y1 - 5));
    });
  }
  
  // GPS & Orientation
  function enableGPS() {
    if (!navigator.geolocation) {
      alert('Geolocation not supported');
      return;
    }
    
    const options = { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 };
    
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        gpsLat = pos.coords.latitude;
        gpsLon = pos.coords.longitude;
        gpsAcc = pos.coords.accuracy;
        updateGPSDisplay();
        
        gpsWatchId = navigator.geolocation.watchPosition(
          (pos) => {
            gpsLat = pos.coords.latitude;
            gpsLon = pos.coords.longitude;
            gpsAcc = pos.coords.accuracy;
            updateGPSDisplay();
          },
          (err) => console.error('[GPS] error:', err),
          options
        );
      },
      (err) => alert(`GPS Error: ${err.message}`),
      options
    );
    
    // Request orientation permission (iOS requires explicit permission)
    if (typeof DeviceOrientationEvent !== 'undefined') {
      if (typeof DeviceOrientationEvent.requestPermission === 'function') {
        DeviceOrientationEvent.requestPermission()
          .then(response => {
            if (response === 'granted') {
              window.addEventListener('deviceorientation', handleOrientation);
            }
          })
          .catch(console.error);
      } else {
        window.addEventListener('deviceorientation', handleOrientation);
      }
    }
  }
  
  function handleOrientation(event) {
    heading = event.webkitCompassHeading || event.alpha;
    pitch = event.beta;
    roll = event.gamma;
    updateHeadingDisplay();
  }
  
  function updateGPSDisplay() {
    if (gpsLat && gpsLon) {
      document.getElementById('gpsIndicator').className = 'gps-indicator active';
      document.getElementById('gpsText').textContent = 
        `GPS: ${gpsLat.toFixed(6)}, ${gpsLon.toFixed(6)} (±${gpsAcc.toFixed(0)}m)`;
      updateMapObserver();
    }
  }
  
  function updateHeadingDisplay() {
    const manualHeading = document.getElementById('manualHeading').value;
    const displayHeading = manualHeading || heading;
    if (displayHeading !== undefined && displayHeading !== null) {
      document.getElementById('headingDisplay').textContent = displayHeading.toFixed(0) + '°';
      document.getElementById('headingSource').textContent = manualHeading ? 'Manual' : 'Compass';
    }
  }
  
  // Send observations with bearing calculation
  function sendObservations(boxes) {
    if (!gpsLat || !gpsLon) return;
    
    const manualHeading = document.getElementById('manualHeading').value;
    const currentHeading = manualHeading ? parseFloat(manualHeading) : (heading || 0);
    hfov = parseFloat(document.getElementById('hfovInput').value) || 63;
    const vfov = hfov * (overlay.height / overlay.width);
    
    boxes.forEach((box, index) => {
      // Calculate bearing from bbox center
      const cx = (box.x1n + box.x2n) / 2;
      const offsetNorm = (cx - 0.5) / 0.5;  // -1 to 1
      const offsetRad = Math.atan(offsetNorm * Math.tan((hfov / 2) * Math.PI / 180));
      const bearing = (currentHeading + (offsetRad * 180 / Math.PI) + 360) % 360;
      
      socket.emit('observation', {
        ts_utc: new Date().toISOString(),
        device_id: deviceId,
        obs_lat: gpsLat,
        obs_lon: gpsLon,
        obs_acc_m: gpsAcc,
        heading_deg: currentHeading,
        hfov_deg: hfov,
        vfov_deg: vfov,
        box: { x1n: box.x1n, y1n: box.y1n, x2n: box.x2n, y2n: box.y2n },
        class: box.label,
        conf: box.conf,
        track_id: box.track_id,
        pitch_deg: pitch,
        roll_deg: roll
      });
    });
  }
  
  // Map initialization
  function initMap() {
    map = L.map('map').setView([0, 0], 2);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: 'OpenStreetMap'
    }).addTo(map);
    bearingLayer = L.layerGroup().addTo(map);
  }
  
  function updateMapObserver() {
    if (!map || !gpsLat || !gpsLon) return;
    
    if (observerMarker) {
      observerMarker.setLatLng([gpsLat, gpsLon]);
    } else {
      observerMarker = L.circleMarker([gpsLat, gpsLon], {
        radius: 8,
        fillColor: '#0284c7',
        color: '#fff',
        weight: 2,
        opacity: 1,
        fillOpacity: 0.8
      }).addTo(map);
      map.setView([gpsLat, gpsLon], 15);
    }
  }
  
  // Load detections from API
  function loadDetections() {
    fetch('/api/detections?limit=50')
      .then(r => r.json())
      .then(data => {
        const tbody = document.querySelector('#detectionsTable tbody');
        tbody.innerHTML = '';
        
        bearingLayer.clearLayers();
        
        (data.detections || []).forEach(d => {
          const tr = document.createElement('tr');
          const time = new Date(d.ts_utc).toLocaleTimeString();
          const mode = d.mode || 'bearing';
          const modeClass = `mode-${mode}`;
          
          tr.innerHTML = `
            <td>${time}</td>
            <td>${d.class || 'N/A'}</td>
            <td>${(d.conf || 0).toFixed(2)}</td>
            <td>${d.bearing_deg ? d.bearing_deg.toFixed(1) + '°' : 'N/A'}</td>
            <td class="${modeClass}">${mode}</td>
            <td>${d.est_lat ? d.est_lat.toFixed(6) : 'N/A'}</td>
            <td>${d.est_lon ? d.est_lon.toFixed(6) : 'N/A'}</td>
            <td>${d.err_m ? d.err_m.toFixed(1) : 'N/A'}</td>
          `;
          tbody.appendChild(tr);
          
          // Draw on map
          if (d.obs_lat && d.obs_lon && d.bearing_deg) {
            // If triangulated, use exact position; otherwise show ray
            const endLat = d.est_lat || d.obs_lat + 0.001 * Math.cos(d.bearing_deg * Math.PI / 180);
            const endLon = d.est_lon || d.obs_lon + 0.001 * Math.sin(d.bearing_deg * Math.PI / 180);
            
            L.polyline([[d.obs_lat, d.obs_lon], [endLat, endLon]], {
              color: mode === 'triangulated' ? '#22c55e' : '#fbbf24',
              weight: 2,
              opacity: 0.6
            }).addTo(bearingLayer);
            
            if (d.est_lat && d.est_lon) {
              L.circleMarker([d.est_lat, d.est_lon], {
                radius: 6,
                fillColor: mode === 'triangulated' ? '#22c55e' : '#60a5fa',
                color: '#fff',
                weight: 2,
                opacity: 1,
                fillOpacity: 0.8
              }).bindPopup(`${mode}: ${d.class} ${(d.conf || 0).toFixed(2)}`).addTo(bearingLayer);
            }
          }
        });
      })
      .catch(console.error);
  }
  
  function displayUAVTarget(target) {
    document.getElementById('uavJson').textContent = JSON.stringify(target, null, 2);
  }
  
  // Event listeners
  document.getElementById('startBtn').onclick = async () => {
    if (!stream) {
      // Try rear camera first, then fall back to default
      let success = false;
      try {
        success = await startCamera({
          video: { 
            facingMode: { ideal: 'environment' },  // Prefer rear camera
            width: { ideal: 1280 }, 
            frameRate: { ideal: 30, max: 30 } 
          },
          audio: false
        });
      } catch (err) {
        console.warn('[CAMERA] Rear camera failed, trying default:', err);
      }
      
      if (!success) {
        success = await startCamera({
          video: { width: { ideal: 1280 }, frameRate: { ideal: 30, max: 30 } },
          audio: false
        });
      }
      
      if (!success) return;
    }
    startCapture();
  };
  
  document.getElementById('stopBtn').onclick = stopCapture;
  document.getElementById('gpsBtn').onclick = enableGPS;
  
  deviceSelect.onchange = async () => {
    const deviceId = deviceSelect.value;
    const wasCapturing = capturing;
    if (wasCapturing) stopCapture();
    
    const constraints = deviceId ?
      { video: { deviceId: { exact: deviceId } }, audio: false } :
      { video: { 
          facingMode: { ideal: 'environment' },  // Default to rear camera
          width: { ideal: 1280 }, 
          frameRate: { ideal: 30, max: 30 } 
        }, audio: false };
    
    const success = await startCamera(constraints);
    if (success && wasCapturing) {
      setTimeout(startCapture, 500);
    }
  };
  
  document.getElementById('hfovInput').onchange = () => {
    hfov = parseFloat(document.getElementById('hfovInput').value) || 63;
  };
  
  document.getElementById('manualHeading').onchange = updateHeadingDisplay;
  
  // Initialize
  connectSocket();
  initMap();
  
  // Auto-start camera (if secure context)
  setTimeout(async () => {
    if (isSecure) {
      // Try to start with rear camera
      try {
        await startCamera({
          video: { 
            facingMode: { ideal: 'environment' },  // Prefer rear camera
            width: { ideal: 1280 }, 
            frameRate: { ideal: 30, max: 30 } 
          },
          audio: false
        });
      } catch (err) {
        console.warn('[CAMERA] Rear camera not available, using default:', err);
        // Fallback to default camera
        await startCamera({
          video: { width: { ideal: 1280 }, frameRate: { ideal: 30, max: 30 } },
          audio: false
        });
      }
    }
  }, 500);
  
  // Periodic UAV handoff updates
  setInterval(() => {
    if (socket && socket.connected) {
      fetch('/api/target/current')
        .then(r => r.status === 200 ? r.json() : null)
        .then(target => {
          if (target) displayUAVTarget(target);
        })
        .catch(() => {});
    }
  }, 2000);
  
})();
