import React, { useState } from "react";
import axios from "axios";
import { Link, useNavigate } from "react-router-dom";

export default function Register() {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleRegister = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      await axios.post("http://localhost:8000/api/register", { name, email, password, role: "teacher" });
      alert("Account created. Please login.");
      navigate("/");
    } catch (err) {
      console.error(err);
      alert(err?.response?.data?.message || "Registration failed.");
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

        <h1 className="title">Créer un compte StudentCoach</h1>
        <p className="subtitle">Rejoignez votre espace d'apprentissage personnalisé</p>

        <form className="form" onSubmit={handleRegister}>
          <div className="input-group">
            <span className="icon">👤</span>
            <input required type="text" placeholder="Prénom" value={name} onChange={(e) => setName(e.target.value)} />
          </div>

          <div className="input-group">
            <span className="icon">👤</span>
            <input required type="text" placeholder="Nom" />
          </div>

          <div className="input-group">
            <span className="icon">✉️</span>
            <input required type="email" placeholder="Adresse e-mail" value={email} onChange={(e) => setEmail(e.target.value)} />
          </div>

          <div className="input-group">
            <span className="icon">🔒</span>
            <input required type="password" placeholder="mot de passe" value={password} onChange={(e) => setPassword(e.target.value)} />
          </div>

          <div className="input-group">
            <span className="icon">🔒</span>
            <input required type="password" placeholder="Confirmation de mot de passe" />
          </div>

          <label className="remember">
            <input type="checkbox" /> <span>J'accepte les Conditions d'utilisation et la Politique de confidentialité</span>
          </label>

          <button className="primary" type="submit" disabled={loading}>{loading ? "Création..." : "Créer mon compte"}</button>

          <p className="links">Déjà un compte ? <Link to="/">Se connecter</Link></p>
        </form>
      </div>
    </div>
  );
}
