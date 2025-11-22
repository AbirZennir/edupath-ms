import React from "react";
import { Link } from "react-router-dom";
import "./Landing.css";

export default function Landing() {
  return (
    <div className="landing-page">
      <div className="landing-inner">
        <div className="landing-logo">
          <svg width="96" height="72" viewBox="0 0 64 48" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M32 0L64 16L32 32L0 16L32 0Z" fill="#2196F3"/>
            <path d="M12 20L32 32L52 20" fill="#1976D2"/>
          </svg>
        </div>

        <h1 className="brand">StudentCoach</h1>
        <div className="brand-sub">EduPath-MS</div>
        <p className="tagline">Votre compagnon d'apprentissage personnalisé</p>

        <div className="cta-row">
          <Link className="btn primary" to="/login">Se connecter</Link>
          <Link className="btn outline" to="/register">Créer un compte</Link>
        </div>

        <p className="footer-note">En continuant, vous acceptez nos Conditions d'utilisation et notre Politique de confidentialité.</p>
      </div>
    </div>
  );
}
