from schemas.recommendation import RecommendationRequest, RecommendationResponse, RecommendationCategory, RecommendationItem

class RecommendationService:
    def generate_recommendations(self, request: RecommendationRequest) -> RecommendationResponse:
        score = request.riskScore
        
        # Determine Profile
        if score >= 0.8:
            profile = "Étudiant à haut risque nécessitant un accompagnement intensif"
            level = "Élevé"
            video_priority = 1
            article_priority = 2
            quiz_priority = 1
        elif score >= 0.5:
            profile = "Étudiant nécessitant un accompagnement ciblé"
            level = "Modéré"
            video_priority = 2
            article_priority = 1
            quiz_priority = 2
        else:
            profile = "Étudiant en faible risque — recommandations générales"
            level = "Faible"
            video_priority = 3
            article_priority = 3
            quiz_priority = 3

        # Videos
        video_items = [
            RecommendationItem(
                id="vid_1",
                title="Méthodes d'apprentissage efficaces",
                description="Courte vidéo présentant des techniques d'apprentissage actif et de gestion du temps.",
                url="https://youtube.com/watch?v=example1",
                type="video",
                duration="15 min",
                difficulty="facile",
                priority=video_priority
            ),
            RecommendationItem(
                id="vid_2",
                title="Organiser son temps pour étudier efficacement",
                description="Stratégies de planification et de priorisation des tâches.",
                url="https://youtube.com/watch?v=example2",
                type="video",
                duration="12 min",
                difficulty="facile",
                priority=1 if score >= 0.8 else 2
            )
        ]
        
        videos_cat = RecommendationCategory(
            category="Vidéos",
            icon="Video",
            color="#3B82F6",
            items=video_items
        )

        # Articles
        article_items = [
            RecommendationItem(
                id="art_1",
                title="Techniques de prise de notes",
                description="Article expliquant Cornell Note-Taking et résumé efficace.",
                url="https://example.com/articles/notes",
                type="article",
                duration="5 min read",
                difficulty="facile",
                priority=article_priority
            ),
            RecommendationItem(
                id="art_2",
                title="Gérer l'anxiété avant les examens",
                description="Conseils pratiques pour réduire le stress et améliorer la performance.",
                url="https://example.com/articles/stress",
                type="article",
                duration="7 min read",
                difficulty="moyen",
                priority=1 if score >= 0.8 else 2
            )
        ]

        articles_cat = RecommendationCategory(
            category="Articles",
            icon="Article",
            color="#10B981",
            items=article_items
        )

        # Exercises
        quiz_items = [
            RecommendationItem(
                id="ex_1",
                title="Quiz de révision rapide",
                description="Un quiz de 10 questions pour renforcer les concepts clés.",
                url="/courses/quick-quiz",
                type="quiz",
                duration="10 min",
                difficulty="moyen",
                priority=quiz_priority
            )
        ]

        exercises_cat = RecommendationCategory(
            category="Exercices",
            icon="Exercise",
            color="#F59E0B",
            items=quiz_items
        )

        return RecommendationResponse(
            studentProfile=profile,
            riskLevel=level,
            categories=[videos_cat, articles_cat, exercises_cat]
        )
