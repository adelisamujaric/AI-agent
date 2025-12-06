// ------------------------------------------------------------------
// BACKEND URL
// ------------------------------------------------------------------
const API_FIRST   = "http://localhost:8000/analyze_first_image";
const API_ZOOM    = "http://localhost:8000/analyze_zoom_image";
const API_CONFIRM = "http://localhost:8000/record_violation";
const API_DETECT  = "http://localhost:8000/detect";
const API_REJECT  = "http://localhost:8000/reject_violation";
const API_OK_DETECTION = "http://localhost:8000/record_ok_detection";
// 🆕 Novi API endpointi za učenje
const API_LEARNING_STATS = "http://localhost:8000/learning_stats";
const API_RETRAIN = "http://localhost:8000/retrain_model";


// ------------------------------------------------------------------
// GLOBAL STATE
// ------------------------------------------------------------------
let state = "FIRST";
let currentViolationId = null;
let detectedDriver = null;
let firstImagePath = null;
let secondImagePath = null;
let isOnReservation = false;

// ------------------------------------------------------------------
// LOADING SPINNER
// ------------------------------------------------------------------
function showSpinner() {
    document.getElementById("loadingSpinner").classList.remove("hidden");
}
function hideSpinner() {
    document.getElementById("loadingSpinner").classList.add("hidden");
}

// ------------------------------------------------------------------
// OPEN FILE PICKER
// ------------------------------------------------------------------
function triggerUpload() {
    document.getElementById("imageInput").click();
}

// ------------------------------------------------------------------
// PREVIEW IMAGE ON LEFT (only for FIRST step)
// ------------------------------------------------------------------
document.getElementById("imageInput").addEventListener("change", function () {
    let file = this.files[0];
    if (!file) return;

    let img = document.getElementById("previewImage");
    img.src = URL.createObjectURL(file);
    img.style.display = "block";

    if (state === "ZOOM") {
        let btn = document.getElementById("actionButton");
        btn.textContent = "🔎 Detektuj tablicu";
        btn.style.background = "#ff5e00";
    }
});

// ------------------------------------------------------------------
// MAIN BUTTON HANDLER
// ------------------------------------------------------------------
async function analyzeOrZoom() {
    let fileInput = document.getElementById("imageInput");
    let file = fileInput.files[0];

    // Ako smo u ZOOM modu i nema file-a, otvori file picker
    if (state === "ZOOM" && !file) {
        triggerUpload();
        return;
    }

    if (!file) {
        alert("Odaberi sliku!");
        return;
    }

    if (state === "FIRST") {
        await analyzeFirstImage(file);
    } else if (state === "ZOOM") {
        await analyzeZoomImage(file);
    }
}

// ------------------------------------------------------------------
// 1️⃣ ANALYZE FIRST IMAGE
// ------------------------------------------------------------------
async function analyzeFirstImage(file) {
    const formData = new FormData();
    formData.append("file", file);

    showSpinner();

    let res = await fetch(API_FIRST, { method: "POST", body: formData });
    let data = await res.json();

    await showFirstDetection(file);
    await drawDetectionsOnImage("canvas1", "firstImage", file);

    hideSpinner();

    if (data.status === "OK") {
        showMessage(data.message || "Nema prekršaja ✔", "green");
        enableConfirmButtons();
        return;
    }

    if (data.status === "NEEDS_ZOOM") {
        currentViolationId = data.prekrsaj_id;
        isOnReservation = data.on_reservation || false;
        showMessage(data.message, "orange");

        let btn = document.getElementById("actionButton");
        btn.textContent = "📸 Učitaj bližu sliku";
        btn.style.background = "#ff9600";

        state = "ZOOM";

        document.getElementById("imageInput").value = "";
        document.getElementById("previewImage").style.display = "none";
    }
}

async function analyzeZoomImage(file) {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("on_reservation", isOnReservation);
    formData.append("prekrsaj_id", currentViolationId);

    showSpinner();

    let res = await fetch(API_ZOOM, { method: "POST", body: formData });
    let data = await res.json();

    await showSecondDetection(file);
    await drawDetectionsOnImage("canvas2", "secondImage", file);

    hideSpinner();

    if (data.status === "NO_PLATE") {
        showMessage("Tablica nije pronađena ❌", "red");
        return;
    }

    if (data.status === "NO_DRIVER") {
        showMessage(`Tablica: ${data.plate} – vozač nije u bazi ❌`, "red");
        return;
    }

    // Vozač ima rezervaciju - parkiranje OK
    if (data.status === "OK_WITH_RESERVATION") {
        showMessage(data.message, "green");
        showDriverCard(
            data.vozac,
            "Parkiranje na rezervaciji",
            "0 (Dozvoljeno)"
        );
        enableConfirmButtons();

        let btn = document.getElementById("actionButton");
        btn.textContent = "🔍 Analiziraj";
        btn.style.background = "#00a86b";
        state = "FIRST";
        return;
    }

    if (data.status === "READY_TO_CONFIRM") {
        detectedDriver = data.vozac;
        firstImagePath = data.slika1;
        secondImagePath = data.slika2;

        showDriverCard(
            data.vozac,
            data.prekrsaj_opis,
            data.prekrsaj_kazna
        );

        enableConfirmButtons();
    }

    let btn = document.getElementById("actionButton");
    btn.textContent = "🔍 Analiziraj";
    btn.style.background = "#00a86b";

    state = "FIRST";
}

// ------------------------------------------------------------------
// CONFIRM VIOLATION
// ------------------------------------------------------------------

async function confirmViolation() {
    console.log("confirmViolation pozvana!");
    console.log("State:", state);
    console.log("currentViolationId:", currentViolationId);
    console.log("detectedDriver:", detectedDriver);

    // 🆕 Scenario 1: Nema prekršaja - samo potvrdi OK detekciju
    if (!currentViolationId || !detectedDriver) {
        console.log("Nema prekršaja - čuvam OK detekciju");

        showSpinner();

        try {
            const formData = new FormData();
            formData.append("image_path", "backend/uploads/first_image.jpg");

            let res = await fetch(API_OK_DETECTION, {
                method: 'POST',
                body: formData
            });

            let data = await res.json();
            console.log("Odgovor:", data);

            hideSpinner();
            //alert("Potvrđeno: Detekcija bez prekršaja sačuvana za učenje.");
            resetUI();
            checkLearningStats();
        } catch (err) {
            console.error("Greška:", err);
            hideSpinner();
            alert("Greška prilikom čuvanja.");
        }

        return;
    }

    // Scenario 2: Ima prekršaj - standardni workflow
    console.log("Ima prekršaj, šaljem u bazu...");

    const payload = {
        vozac_id: detectedDriver.vozac_id,
        prekrsaj_id: currentViolationId,
        slika1: firstImagePath,
        slika2: secondImagePath
    };

    console.log("Payload:", payload);

    showSpinner();

    try {
        let res = await fetch(API_CONFIRM, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        let data = await res.json();
        console.log("Odgovor:", data);

        hideSpinner();
        resetUI();
        checkLearningStats();
    } catch (err) {
        console.error("Greška:", err);
        hideSpinner();
        alert("Greška prilikom evidencije.");
    }
}

// ------------------------------------------------------------------
// REJECT VIOLATION - POJEDNOSTAVLJENO (uvijek radi!)
// ------------------------------------------------------------------
async function rejectViolation() {
    console.log("🔴 rejectViolation() pozvana");
    console.log("State:", state);
    console.log("firstImagePath:", firstImagePath);
    console.log("secondImagePath:", secondImagePath);

    showSpinner();

    try {
        // UVIJEK šalji first image (čak i ako ima second)
        const formData = new FormData();
        formData.append("image_path", firstImagePath || "backend/uploads/first_image.jpg");

        // Ako ima i second image, pošalji i njega
        if (secondImagePath) {
            formData.append("second_image_path", secondImagePath);
        }

        let res = await fetch("http://localhost:8000/reject_detection", {
            method: 'POST',
            body: formData
        });

        let data = await res.json();
        console.log("✅ Odbačeno:", data);

        hideSpinner();
        showMessage(`❌ Detekcija odbačena (${data.count} slika)`, "orange");

        await new Promise(resolve => setTimeout(resolve, 1000));

    } catch (err) {
        console.error("❌ Greška:", err);
        hideSpinner();
        showMessage("Greška pri odbacivanju", "red");
    }

    resetUI();
    checkLearningStats();
}

// ------------------------------------------------------------------
// UI HELPERS
// ------------------------------------------------------------------
function showMessage(text, color) {
    document.getElementById("resultsText").innerHTML =
        `<p style="color:${color};"><b>${text}</b></p>`;
}

function resetAfterOK() {
    let btn = document.getElementById("actionButton");
    btn.textContent = "🔍 Analiziraj";
    btn.style.background = "#00a86b";
    state = "FIRST";
}

function enableConfirmButtons() {
    document.querySelector(".confirm").onclick = confirmViolation;
    document.querySelector(".reject").onclick = rejectViolation;
}

function resetUI() {
    state = "FIRST";
    currentViolationId = null;
    detectedDriver = null;

    document.getElementById("resultsText").innerHTML = "<p>Još nema rezultata.</p>";

    ["firstImage", "secondImage", "previewImage"].forEach(id => {
        let el = document.getElementById(id);
        el.src = "";
        el.style.display = "none";
    });

    ["canvas1", "canvas2"].forEach(id => {
        let canvas = document.getElementById(id);
        let ctx = canvas.getContext("2d");
        canvas.width = 0;
        canvas.height = 0;
    });

    let btn = document.getElementById("actionButton");
    btn.textContent = "🔍 Analiziraj";
    btn.style.background = "#00a86b";
}

// ------------------------------------------------------------------
// IMAGE DISPLAY
// ------------------------------------------------------------------
function showFirstDetection(file) {
    return new Promise((resolve) => {
        let img = document.getElementById("firstImage");
        img.onload = () => {
            console.log("✅ Prva slika učitana!");
            resolve();
        };
        img.src = URL.createObjectURL(file);
        img.style.display = "block";

        if (img.complete) {
            img.onload();
        }
    });
}

function showSecondDetection(file) {
    return new Promise((resolve) => {
        let img = document.getElementById("secondImage");
        img.onload = () => {
            console.log("✅ Druga slika učitana!");
            resolve();
        };
        img.src = URL.createObjectURL(file);
        img.style.display = "block";

        if (img.complete) {
            img.onload();
        }
    });
}

// ------------------------------------------------------------------
// DRAW BOUNDING BOXES SA SKALIRANJEM
// ------------------------------------------------------------------
async function drawDetectionsOnImage(canvasId, imgId, file) {
    console.log(`🎨 Crtanje na ${canvasId}...`);

    const formData = new FormData();
    formData.append("file", file);

    let res = await fetch(API_DETECT, { method: "POST", body: formData });
    let data = await res.json();

    console.log("🔍 Detections:", data);

    let img = document.getElementById(imgId);
    let canvas = document.getElementById(canvasId);
    let ctx = canvas.getContext("2d");

    // Postavi canvas da bude iste veličine kao prikazana slika
    let displayWidth = img.width;
    let displayHeight = img.height;

    canvas.width = displayWidth;
    canvas.height = displayHeight;

    console.log(`📐 Image natural size: ${img.naturalWidth}x${img.naturalHeight}`);
    console.log(`📐 Image display size: ${displayWidth}x${displayHeight}`);
    console.log(`📐 Canvas size: ${canvas.width}x${canvas.height}`);

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!data.detections || data.detections.length === 0) {
        console.warn("⚠️ Nema detekcija!");
        return;
    }

    // Izračunaj scale faktore
    let scaleX = displayWidth / img.naturalWidth;
    let scaleY = displayHeight / img.naturalHeight;

    console.log(`📏 Scale factors: X=${scaleX}, Y=${scaleY}`);

    data.detections.forEach((det, idx) => {
        let [x1, y1, x2, y2] = det.box;

        // Skaliraj koordinate
        x1 *= scaleX;
        y1 *= scaleY;
        x2 *= scaleX;
        y2 *= scaleY;

        console.log(`📦 Detection ${idx}: [${x1.toFixed(1)}, ${y1.toFixed(1)}, ${x2.toFixed(1)}, ${y2.toFixed(1)}] - ${det.class}`);

        ctx.strokeStyle = "#ff3b3b";
        ctx.lineWidth = 1;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        ctx.fillStyle = "red";
        ctx.font = "14px Arial";
        ctx.fillText(det.class, x1, y1 - 5);
    });

    console.log("✅ Bounding boxes nacrtani!");
}

// ------------------------------------------------------------------
// DRIVER CARD
// ------------------------------------------------------------------
function showDriverCard(driver, opis, kazna) {
    document.getElementById("resultsText").innerHTML = `
        <div class="card">
            <h3>🪪 Podaci o vozaču</h3>
            <p><b>Ime:</b> ${driver.ime}</p>
            <p><b>Tablica:</b> ${driver.tablica}</p>
            <p><b>Auto:</b> ${driver.auto_tip}</p>
            <p><b>Invalid:</b> ${driver.invalid ? "DA" : "NE"}</p>
            <p><b>Rezervacija:</b> ${driver.rezervacija ? "DA" : "NE"}</p>

            <hr>

            <h3>⚠️ Prekršaj</h3>
            <p><b>Opis:</b> ${opis}</p>
            <p><b>Kazna:</b> ${kazna} KM</p>
        </div>
    `;
}

// ------------------------------------------------------------------
// 🆕 LEARNING SYSTEM - Check stats
// ------------------------------------------------------------------
async function checkLearningStats() {
    try {
        let res = await fetch(API_LEARNING_STATS);
        let data = await res.json();

        let info = document.getElementById("learningInfo");
        info.innerHTML = `
            ✅ Potvrđene slike: <b>${data.confirmed_images}</b><br>
            ❌ Odbijene (first): <b>${data.rejected_first}</b><br>
            ❌ Odbijene (zoom): <b>${data.rejected_zoom}</b><br>
            <br>
            ${data.ready_for_retraining ?
              "🟢 <b>Spremno za retraining!</b>" :
              `🟡 Potrebno još ${10 - data.confirmed_images} slika`}
        `;

        // Omogući dugme ako ima dovoljno podataka
        let btn = document.getElementById("retrainBtn");
        if (btn) {
            btn.disabled = !data.ready_for_retraining;
        }
    } catch (err) {
        console.error("Greška pri učitavanju learning stats:", err);
    }
}

// ------------------------------------------------------------------
// 🆕 LEARNING SYSTEM - Trigger retraining
// ------------------------------------------------------------------
async function triggerRetraining() {
    if (!confirm("Da li sigurno želiš pokrenuti retraining modela? Ovo može potrajati nekoliko minuta.")) {
        return;
    }

    showSpinner();

    try {
        let res = await fetch(API_RETRAIN, { method: "POST" });
        let data = await res.json();

        hideSpinner();

        if (data.status === "SUCCESS") {
            alert(`✅ ${data.message}\n\nPoboljšanje: +${(data.improvement * 100).toFixed(1)}%`);
        } else if (data.status === "NO_IMPROVEMENT") {
            alert(`⚠️ ${data.message}`);
        } else {
            alert(`ℹ️ ${data.message}`);
        }

        // Osvježi statistiku
        checkLearningStats();
    } catch (err) {
        hideSpinner();
        console.error("Greška pri retrainingu:", err);
        alert("Greška prilikom retraininga modela.");
    }
}

// ------------------------------------------------------------------
// INIT - Učitaj statistiku kad se stranica učita
// ------------------------------------------------------------------
window.addEventListener('load', () => {
    checkLearningStats();

    // Osvježavaj statistiku svakih 30 sekundi
    setInterval(checkLearningStats, 30000);
});