import { useState } from 'react';
import { GraduationCap } from 'lucide-react';
import { api } from '../api/client';

export default function LoginPage({ onLogin, onForgotPassword, onSignup }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const result = await api.login({ email, password });
      if (result?.token) {
        onLogin(result.token);
      } else {
        setError('Réponse inattendue du serveur.');
      }
    } catch (err) {
      const msg = err?.message;
      if (msg && msg.includes('401')) {
        setError('Mot de passe incorrect ou utilisateur introuvable.');
      } else {
        setError(msg || 'Connexion impossible.');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-[#F5F7FB] px-4">
      <div className="w-full max-w-md">
        {/* Logo et titre */}
        <div className="text-center mb-8">
          <div className="flex justify-center mb-4">
            <div className="bg-[#2563EB] p-4 rounded-2xl">
              <GraduationCap className="w-12 h-12 text-white" />
            </div>
          </div>
          <h1 className="text-[#1E293B] mb-2">EduPath-MS</h1>
          <p className="text-[#64748B]">Console Enseignant EduPath-MS</p>
        </div>

        {/* Formulaire de connexion */}
        <div className="bg-white rounded-2xl shadow-lg p-8">
          {error && (
            <div className="mb-4 rounded-lg border border-[#FECACA] bg-[#FEF2F2] px-4 py-3 text-[#B91C1C]">
              {error}
            </div>
          )}
          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label htmlFor="email" className="block text-[#334155] mb-2">
                Adresse email
              </label>
              <input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full px-4 py-3 rounded-lg border border-[#E2E8F0] focus:outline-none focus:ring-2 focus:ring-[#2563EB] focus:border-transparent transition"
                placeholder="enseignant@edupath.fr"
                required
              />
            </div>

            <div>
              <label htmlFor="password" className="block text-[#334155] mb-2">
                Mot de passe
              </label>
              <input
                id="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full px-4 py-3 rounded-lg border border-[#E2E8F0] focus:outline-none focus:ring-2 focus:ring-[#2563EB] focus:border-transparent transition"
                placeholder="••••••••"
                required
              />
            </div>

            <div className="text-right">
              <button
                type="button"
                onClick={onForgotPassword}
                className="text-[#2563EB] hover:underline"
              >
                Mot de passe oublié ?
              </button>
            </div>

            <button
              type="submit"
              className="w-full bg-[#2563EB] text-white py-3 rounded-lg hover:bg-[#1E40AF] transition disabled:opacity-70"
              disabled={loading}
            >
              {loading ? 'Connexion...' : 'Se connecter'}
            </button>
          </form>

          <div className="mt-6 text-center border-t border-[#E2E8F0] pt-6">
            <p className="text-[#64748B] mb-4">
              Vous n&apos;avez pas encore de compte ?
            </p>
            <button
              onClick={onSignup}
              className="w-full bg-white border border-[#2563EB] text-[#2563EB] py-3 rounded-lg hover:bg-[#EFF6FF] transition"
            >
              Créer un compte enseignant
            </button>
          </div>

          <div className="mt-6 text-center">
            <p className="text-[#94A3B8]">
              Accès étudiant sur application mobile{' '}
              <span className="text-[#2563EB]">StudentCoach</span>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}


