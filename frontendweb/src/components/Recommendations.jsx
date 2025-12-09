import { Video, FileText, ClipboardCheck, ExternalLink, User, Users } from 'lucide-react';
import Sidebar from './Sidebar';

const individualRecommendations = [
  {
    student: 'Dubois Alexandre',
    avatar: 'AD',
    resources: [
      { id: 1, titre: 'Tutoriel : Les arbres binaires de recherche', type: 'Vidéo', difficulte: 'Moyen', icon: Video },
      { id: 2, titre: 'Exercices : Parcours d\'arbres', type: 'PDF', difficulte: 'Facile', icon: FileText },
      { id: 3, titre: 'Quiz : Structures de données avancées', type: 'Quiz', difficulte: 'Difficile', icon: ClipboardCheck },
    ],
  },
  {
    student: 'Bernard Lucas',
    avatar: 'BL',
    resources: [
      { id: 4, titre: 'Cours de rattrapage : Complexité algorithmique', type: 'Vidéo', difficulte: 'Facile', icon: Video },
      { id: 5, titre: 'Fiche de révision : Big O notation', type: 'PDF', difficulte: 'Facile', icon: FileText },
    ],
  },
  {
    student: 'Roux Thomas',
    avatar: 'RT',
    resources: [
      { id: 6, titre: 'Projet pratique : Implémentation d\'un tri rapide', type: 'PDF', difficulte: 'Moyen', icon: FileText },
      { id: 7, titre: 'Quiz de diagnostic : Algorithmes de tri', type: 'Quiz', difficulte: 'Moyen', icon: ClipboardCheck },
    ],
  },
];

const collectiveRecommendations = [
  {
    id: 1,
    titre: 'Revoir le chapitre 3 : Structures de données',
    description: '25% des étudiants ont des difficultés sur ce chapitre. Une séance de révision collective est recommandée.',
    studentsAffected: 12,
    priority: 'haute',
  },
  {
    id: 2,
    titre: 'Proposer une séance de tutorat sur les graphes',
    description: '8 étudiants pourraient bénéficier d\'un accompagnement personnalisé sur les algorithmes de graphes.',
    studentsAffected: 8,
    priority: 'moyenne',
  },
  {
    id: 3,
    titre: 'Ajouter des exercices pratiques sur la récursivité',
    description: 'Le taux de réussite sur les exercices de récursivité est de 62%. Des exercices supplémentaires sont nécessaires.',
    studentsAffected: 18,
    priority: 'haute',
  },
  {
    id: 4,
    titre: 'Organiser un atelier de programmation en binôme',
    description: 'Pour améliorer l\'engagement et la collaboration entre étudiants.',
    studentsAffected: 45,
    priority: 'basse',
  },
];

const difficulteColors = {
  'Facile': 'bg-[#DCFCE7] text-[#22C55E]',
  'Moyen': 'bg-[#FED7AA] text-[#F97316]',
  'Difficile': 'bg-[#FEE2E2] text-[#EF4444]',
};

const priorityColors = {
  'haute': 'border-[#EF4444]',
  'moyenne': 'border-[#F97316]',
  'basse': 'border-[#22C55E]',
};

export default function Recommendations({ onNavigate, onLogout }) {
  return (
    <div className="flex">
      <Sidebar currentPage="recommendations" onNavigate={onNavigate} onLogout={onLogout} />
      
      <main className="flex-1 p-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-[#1E293B] mb-2">Recommandations pédagogiques</h1>
          <p className="text-[#64748B]">
            Ressources personnalisées générées par RecoBuilder pour optimiser l&apos;apprentissage
          </p>
        </div>

        {/* Recommandations collectives */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-6">
            <Users className="w-6 h-6 text-[#2563EB]" />
            <h2 className="text-[#1E293B]">Recommandations collectives</h2>
          </div>

          <div className="grid gap-4">
            {collectiveRecommendations.map((reco) => (
              <div
                key={reco.id}
                className={`bg-white rounded-2xl p-6 shadow-sm border-l-4 ${priorityColors[reco.priority]}`}
              >
                <div className="flex items-start justify-between mb-3">
                  <div className="flex-1">
                    <h3 className="text-[#1E293B] mb-2">{reco.titre}</h3>
                    <p className="text-[#64748B] mb-3">{reco.description}</p>
                    <div className="flex items-center gap-2">
                      <span className="text-[#94A3B8]">
                        {reco.studentsAffected} étudiant{reco.studentsAffected > 1 ? 's' : ''} concerné{reco.studentsAffected > 1 ? 's' : ''}
                      </span>
                      <span className="text-[#94A3B8]">•</span>
                      <span className={`${
                        reco.priority === 'haute' ? 'text-[#EF4444]' : 
                        reco.priority === 'moyenne' ? 'text-[#F97316]' : 
                        'text-[#22C55E]'
                      }`}>
                        Priorité {reco.priority}
                      </span>
                    </div>
                  </div>
                  <button className="px-4 py-2 bg-[#2563EB] text-white rounded-lg hover:bg-[#1E40AF] transition whitespace-nowrap ml-4">
                    Planifier l&apos;action
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Recommandations individuelles */}
        <div>
          <div className="flex items-center gap-3 mb-6">
            <User className="w-6 h-6 text-[#2563EB]" />
            <h2 className="text-[#1E293B]">Recommandations individuelles</h2>
          </div>

          <div className="space-y-6">
            {individualRecommendations.map((student, idx) => (
              <div key={idx} className="bg-white rounded-2xl p-6 shadow-sm">
                {/* En-tête étudiant */}
                <div className="flex items-center gap-3 mb-4 pb-4 border-b border-[#F1F5F9]">
                  <div className="w-10 h-10 rounded-full bg-[#2563EB] flex items-center justify-center text-white">
                    {student.avatar}
                  </div>
                  <div>
                    <h3 className="text-[#1E293B]">{student.student}</h3>
                    <p className="text-[#94A3B8]">{student.resources.length} ressources recommandées</p>
                  </div>
                </div>

                {/* Ressources */}
                <div className="grid gap-3">
                  {student.resources.map((resource) => {
                    const Icon = resource.icon;
                    return (
                      <div
                        key={resource.id}
                        className="border border-[#E2E8F0] rounded-lg p-4 hover:border-[#2563EB] hover:bg-[#F8FAFC] transition"
                      >
                        <div className="flex items-start justify-between">
                          <div className="flex items-start gap-3 flex-1">
                            <div className="bg-[#EFF6FF] p-2 rounded-lg">
                              <Icon className="w-5 h-5 text-[#2563EB]" />
                            </div>
                            <div className="flex-1">
                              <h4 className="text-[#1E293B] mb-1">{resource.titre}</h4>
                              <div className="flex items-center gap-3">
                                <span className="text-[#64748B]">{resource.type}</span>
                                <span className={`px-2 py-1 rounded-full ${difficulteColors[resource.difficulte]}`}>
                                  {resource.difficulte}
                                </span>
                              </div>
                            </div>
                          </div>
                          <button className="flex items-center gap-2 px-4 py-2 bg-[#2563EB] text-white rounded-lg hover:bg-[#1E40AF] transition whitespace-nowrap ml-4">
                            <ExternalLink className="w-4 h-4" />
                            Ouvrir dans le LMS
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        </div>
      </main>
    </div>
  );
}


