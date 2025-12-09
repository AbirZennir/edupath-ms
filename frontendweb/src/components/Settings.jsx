import { useState } from 'react';
import { User, Link as LinkIcon, Bell, Check } from 'lucide-react';
import Sidebar from './Sidebar';

export default function Settings({ onNavigate, onLogout }) {
  const [activeTab, setActiveTab] = useState('profile');
  const [moodleConnected, setMoodleConnected] = useState(true);
  const [canvasConnected, setCanvasConnected] = useState(false);
  const [notifications, setNotifications] = useState({
    alertesRisque: true,
    recommandations: true,
    rapportsHebdo: true,
    misesAJourModele: false,
  });

  const handleNotificationToggle = (key) => {
    setNotifications(prev => ({
      ...prev,
      [key]: !prev[key],
    }));
  };

  return (
    <div className="flex">
      <Sidebar currentPage="settings" onNavigate={onNavigate} onLogout={onLogout} />
      
      <main className="flex-1 p-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-[#1E293B] mb-2">Paramètres</h1>
          <p className="text-[#64748B]">
            Gérez votre profil, vos intégrations et vos préférences de notification
          </p>
        </div>

        {/* Onglets */}
        <div className="flex gap-4 border-b border-[#E2E8F0] mb-8">
          <button
            onClick={() => setActiveTab('profile')}
            className={`pb-3 px-2 transition flex items-center gap-2 ${
              activeTab === 'profile'
                ? 'text-[#2563EB] border-b-2 border-[#2563EB]'
                : 'text-[#64748B] hover:text-[#334155]'
            }`}
          >
            <User className="w-4 h-4" />
            Profil enseignant
          </button>
          <button
            onClick={() => setActiveTab('integrations')}
            className={`pb-3 px-2 transition flex items-center gap-2 ${
              activeTab === 'integrations'
                ? 'text-[#2563EB] border-b-2 border-[#2563EB]'
                : 'text-[#64748B] hover:text-[#334155]'
            }`}
          >
            <LinkIcon className="w-4 h-4" />
            Intégrations LMS
          </button>
          <button
            onClick={() => setActiveTab('notifications')}
            className={`pb-3 px-2 transition flex items-center gap-2 ${
              activeTab === 'notifications'
                ? 'text-[#2563EB] border-b-2 border-[#2563EB]'
                : 'text-[#64748B] hover:text-[#334155]'
            }`}
          >
            <Bell className="w-4 h-4" />
            Notifications
          </button>
        </div>

        {/* Contenu des onglets */}
        {activeTab === 'profile' && (
          <div className="max-w-2xl">
            <div className="bg-white rounded-2xl p-6 shadow-sm mb-6">
              <h3 className="text-[#1E293B] mb-6">Informations personnelles</h3>
              
              {/* Avatar */}
              <div className="flex items-center gap-6 mb-8">
                <div className="w-20 h-20 rounded-full bg-[#2563EB] flex items-center justify-center text-white text-2xl">
                  MP
                </div>
                <div>
                  <button className="px-4 py-2 bg-[#EFF6FF] text-[#2563EB] rounded-lg hover:bg-[#DBEAFE] transition mb-2">
                    Changer la photo
                  </button>
                  <p className="text-[#94A3B8]">JPG, PNG ou GIF. Max 2 MB.</p>
                </div>
              </div>

              {/* Formulaire */}
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-[#334155] mb-2">Prénom</label>
                    <input
                      type="text"
                      defaultValue="Marie"
                      className="w-full px-4 py-2 rounded-lg border border-[#E2E8F0] focus:outline-none focus:ring-2 focus:ring-[#2563EB]"
                    />
                  </div>
                  <div>
                    <label className="block text-[#334155] mb-2">Nom</label>
                    <input
                      type="text"
                      defaultValue="Petit"
                      className="w-full px-4 py-2 rounded-lg border border-[#E2E8F0] focus:outline-none focus:ring-2 focus:ring-[#2563EB]"
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-[#334155] mb-2">Email</label>
                  <input
                    type="email"
                    defaultValue="marie.petit@universite.fr"
                    className="w-full px-4 py-2 rounded-lg border border-[#E2E8F0] focus:outline-none focus:ring-2 focus:ring-[#2563EB]"
                  />
                </div>

                <div>
                  <label className="block text-[#334155] mb-2">Département</label>
                  <input
                    type="text"
                    defaultValue="Informatique"
                    className="w-full px-4 py-2 rounded-lg border border-[#E2E8F0] focus:outline-none focus:ring-2 focus:ring-[#2563EB]"
                  />
                </div>

                <div>
                  <label className="block text-[#334155] mb-2">Établissement</label>
                  <input
                    type="text"
                    defaultValue="Université de Paris"
                    className="w-full px-4 py-2 rounded-lg border border-[#E2E8F0] focus:outline-none focus:ring-2 focus:ring-[#2563EB]"
                  />
                </div>
              </div>

              <div className="flex gap-3 mt-6 pt-6 border-t border-[#E2E8F0]">
                <button className="px-6 py-2 bg-[#2563EB] text-white rounded-lg hover:bg-[#1E40AF] transition">
                  Enregistrer les modifications
                </button>
                <button className="px-6 py-2 bg-white border border-[#E2E8F0] text-[#334155] rounded-lg hover:bg-[#F8FAFC] transition">
                  Annuler
                </button>
              </div>
            </div>

            {/* Section sécurité */}
            <div className="bg-white rounded-2xl p-6 shadow-sm">
              <h3 className="text-[#1E293B] mb-6">Sécurité</h3>
              <button className="text-[#2563EB] hover:text-[#1E40AF]">
                Changer le mot de passe
              </button>
            </div>
          </div>
        )}

        {activeTab === 'integrations' && (
          <div className="max-w-3xl">
            <div className="bg-white rounded-2xl p-6 shadow-sm mb-6">
              <h3 className="text-[#1E293B] mb-2">Connectez vos plateformes LMS</h3>
              <p className="text-[#64748B] mb-6">
                Synchronisez vos données avec vos systèmes de gestion de l&apos;apprentissage
              </p>

              <div className="space-y-4">
                {/* Moodle */}
                <div className="border border-[#E2E8F0] rounded-lg p-6">
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-4 flex-1">
                      <div className="w-12 h-12 bg-[#F97316] rounded-lg flex items-center justify-center text-white">
                        M
                      </div>
                      <div className="flex-1">
                        <h4 className="text-[#1E293B] mb-1">Moodle</h4>
                        <p className="text-[#64748B] mb-2">
                          Synchronisez vos cours, étudiants et activités depuis Moodle
                        </p>
                        {moodleConnected && (
                          <div className="flex items-center gap-2 text-[#22C55E]">
                            <Check className="w-4 h-4" />
                            <span>Connecté à moodle.universite.fr</span>
                          </div>
                        )}
                      </div>
                    </div>
                    <button
                      onClick={() => setMoodleConnected(!moodleConnected)}
                      className={`px-6 py-2 rounded-lg transition whitespace-nowrap ${
                        moodleConnected
                          ? 'bg-[#FEE2E2] text-[#EF4444] hover:bg-[#FECACA]'
                          : 'bg-[#2563EB] text-white hover:bg-[#1E40AF]'
                      }`}
                    >
                      {moodleConnected ? 'Déconnecter' : 'Connecter Moodle'}
                    </button>
                  </div>
                  {moodleConnected && (
                    <div className="mt-4 pt-4 border-t border-[#F1F5F9]">
                      <p className="text-[#94A3B8] mb-2">Dernière synchronisation : Il y a 5 minutes</p>
                      <button className="text-[#2563EB] hover:text-[#1E40AF]">
                        Synchroniser maintenant
                      </button>
                    </div>
                  )}
                </div>

                {/* Canvas */}
                <div className="border border-[#E2E8F0] rounded-lg p-6">
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-4 flex-1">
                      <div className="w-12 h-12 bg-[#EF4444] rounded-lg flex items-center justify-center text-white">
                        C
                      </div>
                      <div className="flex-1">
                        <h4 className="text-[#1E293B] mb-1">Canvas LMS</h4>
                        <p className="text-[#64748B] mb-2">
                          Intégrez vos données Canvas pour une analyse complète
                        </p>
                        {canvasConnected && (
                          <div className="flex items-center gap-2 text-[#22C55E]">
                            <Check className="w-4 h-4" />
                            <span>Connecté à canvas.universite.fr</span>
                          </div>
                        )}
                      </div>
                    </div>
                    <button
                      onClick={() => setCanvasConnected(!canvasConnected)}
                      className={`px-6 py-2 rounded-lg transition whitespace-nowrap ${
                        canvasConnected
                          ? 'bg-[#FEE2E2] text-[#EF4444] hover:bg-[#FECACA]'
                          : 'bg-[#2563EB] text-white hover:bg-[#1E40AF]'
                      }`}
                    >
                      {canvasConnected ? 'Déconnecter' : 'Connecter Canvas'}
                    </button>
                  </div>
                </div>
              </div>
            </div>

            {/* Informations OAuth */}
            <div className="bg-[#EFF6FF] border border-[#93C5FD] rounded-2xl p-6">
              <h4 className="text-[#1E293B] mb-2">Authentification sécurisée OAuth 2.0</h4>
              <p className="text-[#64748B]">
                Toutes les connexions utilisent le protocole OAuth 2.0 pour garantir la sécurité de vos données.
                Vos identifiants LMS ne sont jamais stockés sur nos serveurs.
              </p>
            </div>
          </div>
        )}

        {activeTab === 'notifications' && (
          <div className="max-w-2xl">
            <div className="bg-white rounded-2xl p-6 shadow-sm">
              <h3 className="text-[#1E293B] mb-2">Préférences de notification</h3>
              <p className="text-[#64748B] mb-6">
                Choisissez les notifications que vous souhaitez recevoir par email
              </p>

              <div className="space-y-4">
                {/* Alertes à risque */}
                <div className="flex items-start justify-between py-4 border-b border-[#F1F5F9]">
                  <div className="flex-1">
                    <h4 className="text-[#1E293B] mb-1">Alertes étudiants à risque</h4>
                    <p className="text-[#64748B]">
                      Recevez une notification lorsqu&apos;un étudiant est identifié comme étant à risque
                    </p>
                  </div>
                  <button
                    onClick={() => handleNotificationToggle('alertesRisque')}
                    className={`relative w-12 h-6 rounded-full transition ${
                      notifications.alertesRisque ? 'bg-[#2563EB]' : 'bg-[#E2E8F0]'
                    }`}
                  >
                    <div
                      className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${
                        notifications.alertesRisque ? 'translate-x-7' : 'translate-x-1'
                      }`}
                    ></div>
                  </button>
                </div>

                {/* Recommandations */}
                <div className="flex items-start justify-between py-4 border-b border-[#F1F5F9]">
                  <div className="flex-1">
                    <h4 className="text-[#1E293B] mb-1">Nouvelles recommandations</h4>
                    <p className="text-[#64748B]">
                      Soyez informé lorsque de nouvelles recommandations pédagogiques sont générées
                    </p>
                  </div>
                  <button
                    onClick={() => handleNotificationToggle('recommandations')}
                    className={`relative w-12 h-6 rounded-full transition ${
                      notifications.recommandations ? 'bg-[#2563EB]' : 'bg-[#E2E8F0]'
                    }`}
                  >
                    <div
                      className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${
                        notifications.recommandations ? 'translate-x-7' : 'translate-x-1'
                      }`}
                    ></div>
                  </button>
                </div>

                {/* Rapports hebdomadaires */}
                <div className="flex items-start justify-between py-4 border-b border-[#F1F5F9]">
                  <div className="flex-1">
                    <h4 className="text-[#1E293B] mb-1">Rapports hebdomadaires</h4>
                    <p className="text-[#64748B]">
                      Recevez un résumé hebdomadaire de l&apos;activité et des performances de vos classes
                    </p>
                  </div>
                  <button
                    onClick={() => handleNotificationToggle('rapportsHebdo')}
                    className={`relative w-12 h-6 rounded-full transition ${
                      notifications.rapportsHebdo ? 'bg-[#2563EB]' : 'bg-[#E2E8F0]'
                    }`}
                  >
                    <div
                      className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${
                        notifications.rapportsHebdo ? 'translate-x-7' : 'translate-x-1'
                      }`}
                    ></div>
                  </button>
                </div>

                {/* Mises à jour modèle */}
                <div className="flex items-start justify-between py-4">
                  <div className="flex-1">
                    <h4 className="text-[#1E293B] mb-1">Mises à jour du modèle PathPredictor</h4>
                    <p className="text-[#64748B]">
                      Notifications techniques sur les mises à jour et l&apos;entraînement du modèle IA
                    </p>
                  </div>
                  <button
                    onClick={() => handleNotificationToggle('misesAJourModele')}
                    className={`relative w-12 h-6 rounded-full transition ${
                      notifications.misesAJourModele ? 'bg-[#2563EB]' : 'bg-[#E2E8F0]'
                    }`}
                  >
                    <div
                      className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${
                        notifications.misesAJourModele ? 'translate-x-7' : 'translate-x-1'
                      }`}
                    ></div>
                  </button>
                </div>
              </div>

              <div className="mt-6 pt-6 border-t border-[#E2E8F0]">
                <button className="px-6 py-2 bg-[#2563EB] text-white rounded-lg hover:bg-[#1E40AF] transition">
                  Enregistrer les préférences
                </button>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}


