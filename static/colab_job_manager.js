/**
 * Colab Job Manager
 * ─────────────────
 * - /dub_colab এ job submit করে
 * - job_id localStorage-এ সেভ রাখে
 * - SSE দিয়ে real-time status দেখায়
 * - Page reload করলেও job track থাকে
 *
 * Usage: app.py templates-এ include করো
 * <script src="/static/colab_job_manager.js"></script>
 */

const ColabJobManager = (() => {
  const LS_KEY = "colab_jobs_v1";

  // ─── LocalStorage helpers ──────────────────────────────────────
  function lsLoad() {
    try { return JSON.parse(localStorage.getItem(LS_KEY) || "{}"); }
    catch { return {}; }
  }

  function lsSave(store) {
    try { localStorage.setItem(LS_KEY, JSON.stringify(store)); }
    catch {}
  }

  function saveJob(job) {
    const store = lsLoad();
    store[job.job_id] = { ...store[job.job_id], ...job, saved_at: Date.now() };
    lsSave(store);
  }

  function loadJob(job_id) {
    return lsLoad()[job_id] || null;
  }

  function allJobs() {
    const store = lsLoad();
    return Object.values(store).sort((a, b) => (b.created_at || 0) - (a.created_at || 0));
  }

  function removeJob(job_id) {
    const store = lsLoad();
    delete store[job_id];
    lsSave(store);
  }

  // ─── Submit job ────────────────────────────────────────────────
  async function submitJob({ videoUrl, segments, speakerVoices = {}, language = "Bengali", geminiKeys = [] }) {
    const res = await fetch("/dub_colab", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        video_url:      videoUrl,
        segments,
        speaker_voices: speakerVoices,
        language,
        gemini_keys:    geminiKeys,
      }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || "Submit failed");
    }

    const data = await res.json();
    const job  = {
      job_id:        data.job_id,
      status:        "queued",
      created_at:    Date.now() / 1000,
      video_url:     videoUrl,
      segment_count: segments.length,
      output_url:    null,
      error:         null,
      logs:          [],
    };
    saveJob(job);
    return job;
  }

  // ─── Poll একবার ────────────────────────────────────────────────
  async function fetchStatus(job_id) {
    const res = await fetch(`/colab_job/${job_id}`);
    if (!res.ok) return null;
    const data = await res.json();
    saveJob(data);   // localStorage update
    return data;
  }

  // ─── SSE watch ────────────────────────────────────────────────
  /**
   * job_id track করো, callbacks দিয়ে UI update করো।
   * Returns: stop() function
   */
  function watchJob(job_id, { onStatus, onLog, onDone, onError } = {}) {
    // আগে localStorage থেকে check করো — already done?
    const cached = loadJob(job_id);
    if (cached && ["complete", "error"].includes(cached.status)) {
      onDone?.(cached);
      return () => {};
    }

    const es = new EventSource(`/colab_job_sse/${job_id}`);

    es.addEventListener("status", (e) => {
      const data = JSON.parse(e.data);
      saveJob({ job_id, ...data });
      onStatus?.(data);
    });

    es.addEventListener("log", (e) => {
      const { entries } = JSON.parse(e.data);
      const stored = loadJob(job_id) || { job_id, logs: [] };
      stored.logs  = [...(stored.logs || []), ...entries];
      saveJob(stored);
      onLog?.(entries);
    });

    es.addEventListener("done", (e) => {
      const data = JSON.parse(e.data);
      saveJob({ job_id, ...data });
      onDone?.(data);
      es.close();
    });

    es.addEventListener("error", (e) => {
      try {
        const data = JSON.parse(e.data);
        onError?.(data);
      } catch {
        onError?.({ msg: "SSE connection error" });
      }
      es.close();
    });

    es.onerror = () => {
      // SSE drop হলে 5s পরে retry poll
      setTimeout(async () => {
        const job = await fetchStatus(job_id);
        if (job && ["complete", "error"].includes(job.status)) {
          onDone?.(job);
        } else {
          onError?.({ msg: "SSE reconnect failed, retry manually" });
        }
      }, 5000);
    };

    return () => es.close();
  }

  // ─── Worker status ─────────────────────────────────────────────
  async function workerStatus() {
    try {
      const res = await fetch("/colab_worker_status");
      return await res.json();
    } catch {
      return { online: false };
    }
  }

  // ─── UI Helper: job history panel render ──────────────────────
  /**
   * containerId এর element-এ job history render করে
   * Auto-refreshes every 5s
   */
  function renderJobHistory(containerId) {
    const el = document.getElementById(containerId);
    if (!el) return;

    function render() {
      const jobs = allJobs();
      if (!jobs.length) {
        el.innerHTML = `<p class="text-gray-400 text-sm">এখনো কোনো job নেই।</p>`;
        return;
      }

      const statusIcon = { queued:"⏳", processing:"🔄", complete:"✅", error:"❌" };
      const statusColor = {
        queued:     "text-yellow-400",
        processing: "text-blue-400",
        complete:   "text-green-400",
        error:      "text-red-400",
      };

      el.innerHTML = jobs.map(j => {
        const icon  = statusIcon[j.status]  || "❓";
        const color = statusColor[j.status] || "text-gray-400";
        const date  = j.created_at ? new Date(j.created_at * 1000).toLocaleString("bn-BD") : "";
        const size  = j.output_size_mb ? `${j.output_size_mb}MB` : "";

        return `
        <div class="colab-job-card border border-gray-600 rounded p-3 mb-2 bg-gray-800">
          <div class="flex items-center justify-between">
            <span class="font-mono text-xs text-gray-400">${j.job_id}</span>
            <span class="${color} text-sm">${icon} ${j.status}</span>
          </div>
          <div class="text-xs text-gray-500 mt-1">${date} ${size}</div>
          <div class="text-xs text-gray-400 mt-1 truncate">${(j.video_url || "").slice(0, 60)}</div>
          ${j.status === "complete" && j.output_url ? `
            <a href="${j.output_url}" download
               class="mt-2 inline-block px-3 py-1 bg-green-600 text-white text-xs rounded hover:bg-green-500">
              ⬇ Download
            </a>` : ""}
          ${j.status === "error" ? `
            <div class="mt-1 text-red-400 text-xs">${j.error || ""}</div>` : ""}
          ${j.status === "processing" ? `
            <button onclick="ColabJobManager.watchAndShow('${j.job_id}')"
                    class="mt-2 px-2 py-1 bg-blue-600 text-white text-xs rounded hover:bg-blue-500">
              📡 Track
            </button>` : ""}
          <button onclick="ColabJobManager.removeAndRender('${j.job_id}', '${containerId}')"
                  class="mt-2 ml-2 px-2 py-1 bg-gray-600 text-white text-xs rounded hover:bg-gray-500">
            🗑 Remove
          </button>
        </div>`;
      }).join("");
    }

    render();
    return setInterval(render, 5000);
  }

  // ─── Quick helpers for inline HTML buttons ─────────────────────
  function removeAndRender(job_id, containerId) {
    removeJob(job_id);
    renderJobHistory(containerId);
  }

  function watchAndShow(job_id) {
    const logEl = document.getElementById("colab-live-log");
    if (logEl) logEl.innerHTML = "";

    watchJob(job_id, {
      onStatus: (d) => {
        const statusEl = document.getElementById("colab-status-text");
        if (statusEl) statusEl.textContent = `Status: ${d.status}`;
      },
      onLog: (entries) => {
        if (!logEl) return;
        entries.forEach(e => {
          const div = document.createElement("div");
          div.className = "text-xs font-mono " + (e.lvl === "error" ? "text-red-400" : "text-gray-300");
          div.textContent = e.msg;
          logEl.appendChild(div);
          logEl.scrollTop = logEl.scrollHeight;
        });
      },
      onDone: (d) => {
        const statusEl = document.getElementById("colab-status-text");
        if (statusEl) statusEl.textContent = `✅ Complete! ${d.output_size_mb || ""}MB`;
        if (d.output_url) {
          const dlBtn = document.getElementById("colab-download-btn");
          if (dlBtn) {
            dlBtn.href    = d.output_url;
            dlBtn.style.display = "inline-block";
          }
        }
      },
      onError: (e) => {
        if (logEl) {
          const div = document.createElement("div");
          div.className = "text-red-400 text-xs";
          div.textContent = `❌ ${e.msg}`;
          logEl.appendChild(div);
        }
      },
    });
  }

  // ─── Resume pending jobs on page load ─────────────────────────
  function resumePendingJobs() {
    const jobs = allJobs();
    jobs
      .filter(j => ["queued", "processing"].includes(j.status))
      .forEach(j => {
        console.log(`[ColabJobManager] Resuming watch: ${j.job_id}`);
        // Background poll — no UI hookup, just localStorage update
        watchJob(j.job_id, {
          onDone: (d) => console.log(`[ColabJobManager] ${j.job_id} done:`, d.status),
        });
      });
  }

  // Auto-resume on load
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", resumePendingJobs);
  } else {
    resumePendingJobs();
  }

  // ─── Public API ────────────────────────────────────────────────
  return {
    submitJob,
    fetchStatus,
    watchJob,
    watchAndShow,
    workerStatus,
    allJobs,
    loadJob,
    saveJob,
    removeJob,
    removeAndRender,
    renderJobHistory,
  };
})();

// Global expose
window.ColabJobManager = ColabJobManager;
