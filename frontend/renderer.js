const API_URL = "http://localhost:8000";

// --- Navegacion de Vistas ---
document.getElementById('nav-analyze').addEventListener('click', () => switchView('section-analyze', 'nav-analyze'));
document.getElementById('nav-history').addEventListener('click', () => {
    switchView('section-history', 'nav-history');
    loadHistory();
});

function switchView(sectionId, btnId) {
    document.querySelectorAll('.view-section').forEach(el => el.classList.add('hidden'));
    document.querySelectorAll('.nav-btn').forEach(el => el.classList.remove('active'));
    
    document.getElementById(sectionId).classList.remove('hidden');
    document.getElementById(btnId).classList.add('active');
}

// --- Logica de Analisis con el Backend Python ---
document.getElementById('upload-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const name = document.getElementById('patientName').value;
    const datFile = document.getElementById('datFile').files[0];
    const atrFile = document.getElementById('atrFile').files[0];
    const heaFile = document.getElementById('heaFile').files[0];

    const formData = new FormData();
    formData.append('patient_name', name);
    formData.append('dat_file', datFile);
    formData.append('atr_file', atrFile);
    formData.append('hea_file', heaFile);

    const btn = document.getElementById('submit-btn');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');

    btn.disabled = true;
    loading.classList.remove('hidden');
    results.classList.add('hidden');

    try {
        const response = await fetch(`${API_URL}/analyze`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            // Mostrar resultados
            document.getElementById('res-total').textContent = data.total_beats;
            document.getElementById('res-abnormal').textContent = data.abnormal_beats;
            
            const diagBox = document.getElementById('res-diagnosis');
            diagBox.textContent = data.diagnosis;
            
            // Asignar colores segun gravedad
            diagBox.className = 'diagnosis-box'; // reset
            if (data.abnormal_beats === 0) {
                // Success por defecto
            } else if (data.abnormal_beats < (data.total_beats * 0.05)) {
                diagBox.classList.add('warning');
            } else {
                diagBox.classList.add('danger');
            }
            
            // Mostrar graficos si existen
            const plotsSection = document.getElementById('plots-section');
            const plotsContainer = document.getElementById('abnormal-plots');
            plotsContainer.innerHTML = ''; // limpiar anteriores
            
            if (data.plots && data.plots.length > 0) {
                data.plots.forEach(imgBase64 => {
                    const img = document.createElement('img');
                    img.src = imgBase64;
                    img.className = 'ecg-plot-img';
                    plotsContainer.appendChild(img);
                });
                plotsSection.classList.remove('hidden');
            } else {
                plotsSection.classList.add('hidden');
            }
            
            results.classList.remove('hidden');
        } else {
            alert("Error del Motor IA: " + (data.error || data.detail));
        }
    } catch (error) {
        alert("Error de conexión. ¿Está encendido el Servidor Python?");
        console.error(error);
    } finally {
        btn.disabled = false;
        loading.classList.add('hidden');
    }
});

// --- Carga de Historial SQLite ---
async function loadHistory() {
    const tbody = document.getElementById('history-body');
    tbody.innerHTML = "<tr><td colspan='5' style='text-align:center; padding: 2rem;'>Cargando base de datos...</td></tr>";
    
    try {
        const response = await fetch(`${API_URL}/history`);
        const data = await response.json();
        
        tbody.innerHTML = "";
        if (data.length === 0) {
            tbody.innerHTML = "<tr><td colspan='5' style='text-align:center; padding: 2rem; color: #94A3B8;'>No hay registros clínicos en el sistema.</td></tr>";
            return;
        }

        data.forEach(row => {
            const tr = document.createElement('tr');
            
            // Estilizar el numero de anormales
            let abnormalStyle = "color: #10B981; font-weight: bold;";
            if (row.abnormal_beats > 0) abnormalStyle = "color: #EF4444; font-weight: bold;";
            
            tr.innerHTML = `
                <td style="color: #94A3B8;">${row.date}</td>
                <td><strong style="color: white;">${row.patient_name}</strong></td>
                <td>${row.total_beats}</td>
                <td style="${abnormalStyle}">${row.abnormal_beats}</td>
                <td>${row.diagnosis}</td>
            `;
            tbody.appendChild(tr);
        });
    } catch (error) {
        tbody.innerHTML = "<tr><td colspan='5' style='text-align:center; padding: 2rem; color: #EF4444;'>No se pudo conectar a la base de datos (Backend inactivo)</td></tr>";
    }
}
