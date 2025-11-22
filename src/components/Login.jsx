import React, { useState } from "react";
import axios from "axios";
import { Link, useNavigate } from "react-router-dom";

export default function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleLogin = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      
      const res = await axios.post("http://localhost:8000/api/login", { email, password });
      
      localStorage.setItem("token", res.data.token);
      navigate("/dashboard");
    } catch (err) {
      console.error(err);
      alert(err?.response?.data?.message || "Login failed — check credentials or backend.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-page">
      <div className="auth-card">
        <div className="logo-wrap">
          <svg width="64" height="48" viewBox="0 0 64 48" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M32 0L64 16L32 32L0 16L32 0Z" fill="#2196F3"/>
            <path d="M12 20L32 32L52 20" fill="#1976D2"/>
          </svg>
        </div>

        <h1 className="title">Connexion à StudentCoach</h1>
        <p className="subtitle">Accédez à votre espace d'apprentissage personnalisé</p>

        <form className="form" onSubmit={handleLogin}>
          <div className="input-group">
            <span className="icon">✉️</span>
            <input
              required
              type="email"
              placeholder="Adresse e-mail"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </div>

          <div className="input-group">
            <span className="icon">🔒</span>
            <input
              required
              type="password"
              placeholder="Mot de passe"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </div>

          <label className="remember">
            <input type="checkbox" /> <span>Se souvenir de moi</span>
          </label>

          <button className="primary" type="submit" disabled={loading}>
            {loading ? "Connexion..." : "Se connecter"}
          </button>

          <p className="links">
            Pas encore de compte ? <Link to="/register">Créez-en un</Link>
          </p>
          <p className="links">
            <Link to="#">Mot de passe oublié ?</Link>
          </p>
        </form>
      </div>
    </div>
  );
}
