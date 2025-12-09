import { useState } from 'react';
import { GraduationCap, ArrowLeft } from 'lucide-react';
import { api } from '../api/client';

export default function SignupPage({ onSignup, onBackToLogin }) {
  const [formData, setFormData] = useState({
    firstName: '',
    lastName: '',
    email: '',
    password: '',
    confirmPassword: '',
    department: '',
    institution: '',
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    if (formData.password !== formData.confirmPassword) {
      setError('Les mots de passe ne correspondent pas');
      return;
    }

    setLoading(true);
    try {
      const payload = {
        prenom: formData.firstName,
        nom: formData.lastName,
        email: formData.email,
        password: formData.password,
        departement: formData.department,
        institution: formData.institution,
        role: 'PROFESSEUR',
      };
      const result = await api.register(payload);
      if (result?.token) {
        onSignup(result.token);
      } else {
        setError('Réponse inattendue du serveur.');
      }
    } catch (err) {
      setError(err.message || 'Inscription impossible.');
    } finally {
      setLoading(false);
    }
  };

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-[#F5F7FB] px-4 py-8">
      <div className="w-full max-w-2xl">
        {/* Bouton retour */}
        <button
          onClick={onBackToLogin}
          className="flex items-center gap-2 text-[#2563EB] hover:text-[#1E40AF] mb-6"
        >
          <ArrowLeft className="w-5 h-5" />
          Retour à la connexion
        </button>

        {/* Logo et titre */}
        <div className="text-center mb-8">
          <div className="flex justify-center mb-4">
            <div className="bg-[#2563EB] p-4 rounded-2xl">
              <GraduationCap className="w-12 h-12 text-white" />
            </div>
          </div>
          <h1 className="text-[#1E293B] mb-2">Créer un compte enseignant</h1>
          <p className="text-[#64748B]">Rejoignez EduPath-MS et optimisez votre enseignement</p>
        </div>

        {/* Formulaire d'inscription */}
        <div className="bg-white rounded-2xl shadow-lg p-8">
          {error && (
            <div className="mb-4 rounded-lg border border-[#FECACA] bg-[#FEF2F2] px-4 py-3 text-[#B91C1C]">
              {error}
            </div>
          )}
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Nom et Prénom */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label htmlFor="firstName" className="block text-[#334155] mb-2">
                  Prénom *
                </label>
                <input
                  id="firstName"
                  name="firstName"
                  type="text"
                  value={formData.firstName}
                  onChange={handleChange}
                  className="w-full px-4 py-3 rounded-lg border border-[#E2E8F0] focus:outline-none focus:ring-2 focus:ring-[#2563EB] focus:border-transparent transition"
                  placeholder="Marie"
                  required
                />
              </div>
              <div>
                <label htmlFor="lastName" className="block text-[#334155] mb-2">
                  Nom *
                </label>
                <input
                  id="lastName"
                  name="lastName"
                  type="text"
                  value={formData.lastName}
                  onChange={handleChange}
                  className="w-full px-4 py-3 rounded-lg border border-[#E2E8F0] focus:outline-none focus:ring-2 focus:ring-[#2563EB] focus:border-transparent transition"
                  placeholder="Petit"
                  required
                />
              </div>
            </div>

            {/* Email */}
            <div>
              <label htmlFor="email" className="block text-[#334155] mb-2">
                Adresse email professionnelle *
              </label>
              <input
                id="email"
                name="email"
                type="email"
                value={formData.email}
                onChange={handleChange}
                className="w-full px-4 py-3 rounded-lg border border-[#E2E8F0] focus:outline-none focus:ring-2 focus:ring-[#2563EB] focus:border-transparent transition"
                placeholder="enseignant@universite.fr"
                required
              />
            </div>

            {/* Département et Établissement */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label htmlFor="department" className="block text-[#334155] mb-2">
                  Département *
                </label>
                <input
                  id="department"
                  name="department"
                  type="text"
                  value={formData.department}
                  onChange={handleChange}
                  className="w-full px-4 py-3 rounded-lg border border-[#E2E8F0] focus:outline-none focus:ring-2 focus:ring-[#2563EB] focus:border-transparent transition"
                  placeholder="Informatique"
                  required
                />
              </div>
              <div>
                <label htmlFor="institution" className="block text-[#334155] mb-2">
                  Établissement *
                </label>
                <input
                  id="institution"
                  name="institution"
                  type="text"
                  value={formData.institution}
                  onChange={handleChange}
                  className="w-full px-4 py-3 rounded-lg border border-[#E2E8F0] focus:outline-none focus:ring-2 focus:ring-[#2563EB] focus:border-transparent transition"
                  placeholder="Université de Paris"
                  required
                />
              </div>
            </div>

            {/* Mot de passe */}
            <div>
              <label htmlFor="password" className="block text-[#334155] mb-2">
                Mot de passe *
              </label>
              <input
                id="password"
                name="password"
                type="password"
                value={formData.password}
                onChange={handleChange}
                className="w-full px-4 py-3 rounded-lg border border-[#E2E8F0] focus:outline-none focus:ring-2 focus:ring-[#2563EB] focus:border-transparent transition"
                placeholder="********"
                required
              />
              <p className="text-[#94A3B8] mt-1">
                Minimum 8 caractères avec majuscules, minuscules et chiffres
              </p>
            </div>

            {/* Confirmation mot de passe */}
            <div>
              <label htmlFor="confirmPassword" className="block text-[#334155] mb-2">
                Confirmer le mot de passe *
              </label>
              <input
                id="confirmPassword"
                name="confirmPassword"
                type="password"
                value={formData.confirmPassword}
                onChange={handleChange}
                className="w-full px-4 py-3 rounded-lg border border-[#E2E8F0] focus:outline-none focus:ring-2 focus:ring-[#2563EB] focus:border-transparent transition"
                placeholder="********"
                required
              />
            </div>

            {/* Conditions d'utilisation */}
            <div className="flex items-start gap-3">
              <input
                type="checkbox"
                id="terms"
                className="mt-1 w-4 h-4 rounded border-[#E2E8F0] text-[#2563EB] focus:ring-2 focus:ring-[#2563EB]"
                required
              />
              <label htmlFor="terms" className="text-[#64748B]">
                J'accepte les{' '}
                <a href="#" className="text-[#2563EB] hover:underline">
                  conditions d'utilisation
                </a>{' '}
                et la{' '}
                <a href="#" className="text-[#2563EB] hover:underline">
                  politique de confidentialité
                </a>
              </label>
            </div>

            <button
              type="submit"
              className="w-full bg-[#2563EB] text-white py-3 rounded-lg hover:bg-[#1E40AF] transition disabled:opacity-70"
              disabled={loading}
            >
              {loading ? 'Inscription...' : 'Créer mon compte'}
            </button>
          </form>

          <div className="mt-6 text-center">
            <p className="text-[#64748B]">
              Vous avez déjà un compte ?{' '}
              <button onClick={onBackToLogin} className="text-[#2563EB] hover:underline">
                Se connecter
              </button>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
