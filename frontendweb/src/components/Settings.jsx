import { useState } from 'react';
import { User, Bell } from 'lucide-react';
import Sidebar from './Sidebar';
import { api } from '../api/client';

export default function Settings({ onNavigate, onLogout, user }) {
  const [activeTab, setActiveTab] = useState('profile');
  const [notifications, setNotifications] = useState({
    alertesRisque: true,
    recommandations: true,
    rapportsHebdo: true,
    misesAJourModele: false,
  });
  
  // Form state for profile
  const [firstName, setFirstName] = useState(user?.name ? user.name.split(' ')[0] : '');
  const [lastName, setLastName] = useState(user?.name ? user.name.split(' ').slice(1).join(' ') : '');
  const [email, setEmail] = useState(user?.email || '');
  const [department, setDepartment] = useState('Informatique');
  const [institution, setInstitution] = useState('Université de Paris');
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState(null);

  const handleNotificationToggle = (key) => {
    setNotifications(prev => ({
      ...prev,
      [key]: !prev[key],
    }));
  };

  const handleProfileUpdate = async (e) => {
    e.preventDefault();
    setError(null);
    setSuccess(false);
    setLoading(true);
    
    try {
      const token = localStorage.getItem('authToken');
      const updatedData = {
        firstName,
        lastName,
        email,
        department,
        institution,
        name: `${firstName} ${lastName}`.trim()
      };
      
      // Update in backend (if endpoint exists)
      // await api.updateProfile(updatedData, token);
      
      // Update local storage
      const currentUser = JSON.parse(localStorage.getItem('user') || '{}');
      const updatedUser = { ...currentUser, ...updatedData };
      localStorage.setItem('user', JSON.stringify(updatedUser));
      
      setSuccess(true);
      setTimeout(() => setSuccess(false), 3000);
    } catch (err) {
      setError(err?.message || 'Erreur lors de la mise à jour du profil');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex">
      <Sidebar currentPage="settings" onNavigate={onNavigate} onLogout={onLogout} user={user} />
      
      <main className="flex-1 p-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-[#1E293B] mb-2">Paramètres</h1>
          <p className="text-[#64748B]">
            Gérez votre profil et vos préférences de notification
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
                  {user?.name ? user.name.split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2) : 'MP'}
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
                      value={firstName}
                      onChange={(e) => setFirstName(e.target.value)}
                      className="w-full px-4 py-2 rounded-lg border border-[#E2E8F0] focus:outline-none focus:ring-2 focus:ring-[#2563EB]"
                    />
                  </div>
                  <div>
                    <label className="block text-[#334155] mb-2">Nom</label>
                    <input
                      type="text"
                      value={lastName}
                      onChange={(e) => setLastName(e.target.value)}
                      className="w-full px-4 py-2 rounded-lg border border-[#E2E8F0] focus:outline-none focus:ring-2 focus:ring-[#2563EB]"
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-[#334155] mb-2">Email</label>
                  <input
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className="w-full px-4 py-2 rounded-lg border border-[#E2E8F0] focus:outline-none focus:ring-2 focus:ring-[#2563EB]"
                  />
                </div>

                <div>
                  <label className="block text-[#334155] mb-2">Département</label>
                  <input
                    type="text"
                    value={department}
                    onChange={(e) => setDepartment(e.target.value)}
                    className="w-full px-4 py-2 rounded-lg border border-[#E2E8F0] focus:outline-none focus:ring-2 focus:ring-[#2563EB]"
                  />
                </div>

                <div>
                  <label className="block text-[#334155] mb-2">Établissement</label>
                  <input
                    type="text"
                    value={institution}
                    onChange={(e) => setInstitution(e.target.value)}
                    className="w-full px-4 py-2 rounded-lg border border-[#E2E8F0] focus:outline-none focus:ring-2 focus:ring-[#2563EB]"
                  />
                </div>
              </div>

              {/* Success/Error Messages */}
              {success && (
                <div className="mt-4 rounded-lg border border-[#86EFAC] bg-[#F0FDF4] px-4 py-3 text-[#15803D]">
                  Profil mis à jour avec succès!
                </div>
              )}
              {error && (
                <div className="mt-4 rounded-lg border border-[#FECACA] bg-[#FEF2F2] px-4 py-3 text-[#B91C1C]">
                  {error}
                </div>
              )}

              <div className="flex gap-3 mt-6 pt-6 border-t border-[#E2E8F0]">
                <button 
                  onClick={handleProfileUpdate}
                  disabled={loading}
                  className="px-6 py-2 bg-[#2563EB] text-white rounded-lg hover:bg-[#1E40AF] transition disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loading ? 'Enregistrement...' : 'Enregistrer les modifications'}
                </button>
                <button 
                  onClick={() => {
                    setFirstName(user?.name ? user.name.split(' ')[0] : '');
                    setLastName(user?.name ? user.name.split(' ').slice(1).join(' ') : '');
                    setEmail(user?.email || '');
                    setDepartment('Informatique');
                    setInstitution('Université de Paris');
                    setError(null);
                    setSuccess(false);
                  }}
                  disabled={loading}
                  className="px-6 py-2 bg-white border border-[#E2E8F0] text-[#334155] rounded-lg hover:bg-[#F8FAFC] transition disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Annuler
                </button>
              </div>
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


