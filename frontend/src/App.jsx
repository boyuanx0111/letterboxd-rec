import { useState } from "react";
import "./App.css";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

function App() {
  const [file, setFile] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [importSummary, setImportSummary] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const getRecommendations = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);
    setError("");
    setRecommendations([]);
    setImportSummary(null);

    try {
      const response = await fetch(`${API_BASE_URL}/recommend`, {
        method: "POST",
        body: formData,
      });

      const responseText = await response.text();
      let data = {};
      try {
        data = responseText ? JSON.parse(responseText) : {};
      } catch {
        data = {};
      }

      if (!response.ok) {
        const detail = typeof data.detail === "string" ? data.detail : null;
        throw new Error(detail || "Could not generate recommendations.");
      }

      setRecommendations(data.recommendations || []);
      setImportSummary(data.import_summary || null);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="app-shell">
      <header className="app-header">
        <p className="eyebrow">PERSONALISED MOVIE DISCOVERY</p>
        <h1>Find your next favourite film.</h1>
        <p className="intro-copy">
          Upload your Letterboxd ratings export and get recommendations shaped
          by what you enjoy.
        </p>
      </header>

      <section className="upload-panel" aria-labelledby="upload-heading">
        <h2 id="upload-heading">Your ratings</h2>
        <p className="section-copy">Choose the ratings.csv from your Letterboxd data export.</p>

        <label className={`file-picker${loading ? " is-disabled" : ""}`}>
          <span>{file ? file.name : "Choose a CSV file"}</span>
          <input
            type="file"
            accept=".csv"
            disabled={loading}
            onChange={(event) => {
              setFile(event.target.files[0] || null);
              setRecommendations([]);
              setImportSummary(null);
              setError("");
            }}
          />
        </label>

        <button
          className="recommend-button"
          disabled={!file || loading}
          onClick={getRecommendations}
        >
          {loading ? "Matching your films..." : "Get recommendations"}
        </button>

        {loading && (
          <div className="loading-status" role="status" aria-live="polite">
            <p className="loading-copy">Looking up film IDs and ranking your matches.</p>
            <div
              className="loading-track"
              role="progressbar"
              aria-label="Calculating your recommendations"
              aria-valuetext="Calculation in progress"
            >
              <span className="loading-indicator" />
            </div>
          </div>
        )}
        {error && <p className="error-message" role="alert">{error}</p>}
      </section>

      {recommendations.length > 0 && (
        <section className="results-section" aria-labelledby="results-heading">
          <div className="results-heading-row">
            <div>
              <p className="eyebrow">PICKED FOR YOU</p>
              <h2 id="results-heading">Your recommendations</h2>
            </div>
            <span className="result-count">{recommendations.length} films</span>
          </div>

          {importSummary && (
            <p className="import-summary">
              Based on {importSummary.catalogue_rows} of {importSummary.total_rows} ratings
              matched to the catalogue.
            </p>
          )}

          <ol className="recommendation-list">
          {recommendations.map((movie) => (
              <li className="recommendation-card" key={movie.tmdbId}>
                <span className="recommendation-rank">
                  {String(movie.xgb_rank).padStart(2, "0")}
                </span>
                <span className="recommendation-title">{movie.movie_title}</span>
                <a
                  className="tmdb-link"
                  href={`https://www.themoviedb.org/movie/${movie.tmdbId}`}
                  target="_blank"
                  rel="noreferrer"
                  aria-label={`View ${movie.movie_title} on TMDB`}
                >
                  TMDB <span aria-hidden="true">↗</span>
                </a>
              </li>
          ))}
          </ol>
        </section>
      )}
    </main>
  );
}

export default App;
