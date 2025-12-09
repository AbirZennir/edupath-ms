import { useState } from 'react';
import { GraduationCap, ArrowLeft, Mail, Check } from 'lucide-react';

export default function ForgotPasswordPage({ onBackToLogin }) {
  const [email, setEmail] = useState('');
  const [emailSent, setEmailSent] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    // Simuler l'envoi de l'email
    setEmailSent(true);
  };

  if (emailSent) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[#F5F7FB] px-4">
        <div className="w-full max-w-md">
          <button
            onClick={onBackToLogin}
            className="flex items-center gap-2 text-[#2563EB] hover:text-[#1E40AF] mb-6"
          >
            <ArrowLeft className="w-5 h-5" />
            Retour à la connexion
          </button>

          <div className="bg-white rounded-2xl shadow-lg p-8 text-center">
            <div className="flex justify-center mb-6">
              <div className="bg-[#DCFCE7] p-4 rounded-full">
                <Check className="w-12 h-12 text-[#22C55E]" />
              </div>
            </div>

            <h2 className="text-[#1E293B] mb-3">Email envoyé !</h2>
            <p className="text-[#64748B] mb-6">
              Nous avons envoyé un lien de réinitialisation à <strong>{email}</strong>
            </p>

            <div className="bg-[#EFF6FF] border border-[#93C5FD] rounded-lg p-4 mb-6">
              <p className="text-[#334155]">
                Vérifiez votre boîte de réception et cliquez sur le lien pour réinitialiser votre mot de passe.
                Le lien expire dans 24 heures.
              </p>
            </div>

            <button
              onClick={onBackToLogin}
              className="w-full bg-[#2563EB] text-white py-3 rounded-lg hover:bg-[#1E40AF] transition"
            >
              Retour à la connexion
            </button>

            <div className="mt-4">
              <button
                onClick={() => setEmailSent(false)}
                className="text-[#64748B] hover:text-[#334155]"
              >
                Je n&apos;ai pas reçu l&apos;email
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-[#F5F7FB] px-4">
      <div className="w-full max-w-md">
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
          <h1 className="text-[#1E293B] mb-2">Mot de passe oublié ?</h1>
          <p className="text-[#64748B]">
            Pas de problème, nous vous enverrons un lien de réinitialisation
          </p>
        </div>

        {/* Formulaire */}
        <div className="bg-white rounded-2xl shadow-lg p-8">
          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label htmlFor="email" className="block text-[#334155] mb-2">
                Adresse email
              </label>
              <div className="relative">
                <Mail className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-[#94A3B8]" />
                <input
                  id="email"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full pl-12 pr-4 py-3 rounded-lg border border-[#E2E8F0] focus:outline-none focus:ring-2 focus:ring-[#2563EB] focus:border-transparent transition"
                  placeholder="enseignant@edupath.fr"
                  required
                />
              </div>
            </div>

            <button
              type="submit"
              className="w-full bg-[#2563EB] text-white py-3 rounded-lg hover:bg-[#1E40AF] transition"
            >
              Envoyer le lien de réinitialisation
            </button>
          </form>

          <div className="mt-6 text-center">
            <p className="text-[#64748B]">
              Vous vous souvenez de votre mot de passe ?{' '}
              <button onClick={onBackToLogin} className="text-[#2563EB] hover:underline">
                Se connecter
              </button>
            </p>
          </div>
        </div>

        {/* Information de sécurité */}
        <div className="mt-6 bg-[#FEF3C7] border border-[#FDE68A] rounded-lg p-4">
          <p className="text-[#92400E]">
            <strong>Note de sécurité :</strong> Si vous ne recevez pas l&apos;email dans les prochaines minutes,
            vérifiez votre dossier spam ou contactez l&apos;administrateur système.
          </p>
        </div>
      </div>
    </div>
  );
}


