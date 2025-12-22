package com.example.services;

import com.example.dto.RecommendationCategoryDto;
import com.example.dto.RecommendationItemDto;
import com.example.dto.RecommendationRequest;
import com.example.dto.RecommendationResponse;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
public class RecommendationsService {

    public RecommendationResponse generate(RecommendationRequest req) {
        RecommendationResponse resp = new RecommendationResponse();

        double score = req.getRiskScore() == null ? 0.0 : req.getRiskScore();

        // Simple logic to decide profile and risk level.
        String profile;
        String level;
        if (score >= 0.8) {
            profile = "Étudiant à haut risque nécessitant un accompagnement intensif";
            level = "Élevé";
        } else if (score >= 0.5) {
            profile = "Étudiant nécessitant un accompagnement ciblé";
            level = "Modéré";
        } else {
            profile = "Étudiant en faible risque — recommandations générales";
            level = "Faible";
        }

        resp.setStudentProfile(profile);
        resp.setRiskLevel(level);

        List<RecommendationCategoryDto> categories = new ArrayList<>();

        // Videos category
        RecommendationCategoryDto videos = new RecommendationCategoryDto();
        videos.setCategory("Vidéos");
        videos.setIcon("Video");
        videos.setColor("#3B82F6");

        List<RecommendationItemDto> videoItems = new ArrayList<>();

        RecommendationItemDto vid1 = new RecommendationItemDto();
        vid1.setId("vid_1");
        vid1.setTitle("Méthodes d'apprentissage efficaces");
        vid1.setDescription("Courte vidéo présentant des techniques d'apprentissage actif et de gestion du temps.");
        vid1.setUrl("https://youtube.com/watch?v=example1");
        vid1.setType("video");
        vid1.setDuration("15 min");
        vid1.setDifficulty("facile");
        vid1.setPriority(score >= 0.5 ? 1 : 3);
        videoItems.add(vid1);

        RecommendationItemDto vid2 = new RecommendationItemDto();
        vid2.setId("vid_2");
        vid2.setTitle("Organiser son temps pour étudier efficacement");
        vid2.setDescription("Stratégies de planification et de priorisation des tâches.");
        vid2.setUrl("https://youtube.com/watch?v=example2");
        vid2.setType("video");
        vid2.setDuration("12 min");
        vid2.setDifficulty("facile");
        vid2.setPriority(score >= 0.8 ? 1 : 2);
        videoItems.add(vid2);

        videos.setItems(videoItems);
        categories.add(videos);

        // Articles category
        RecommendationCategoryDto articles = new RecommendationCategoryDto();
        articles.setCategory("Articles");
        articles.setIcon("Article");
        articles.setColor("#10B981");

        List<RecommendationItemDto> articleItems = new ArrayList<>();
        RecommendationItemDto art1 = new RecommendationItemDto();
        art1.setId("art_1");
        art1.setTitle("Techniques de prise de notes");
        art1.setDescription("Article expliquant Cornell Note-Taking et résumé efficace.");
        art1.setUrl("https://example.com/articles/notes");
        art1.setType("article");
        art1.setDuration("5 min read");
        art1.setDifficulty("facile");
        art1.setPriority(score >= 0.5 ? 1 : 3);
        articleItems.add(art1);

        RecommendationItemDto art2 = new RecommendationItemDto();
        art2.setId("art_2");
        art2.setTitle("Gérer l'anxiété avant les examens");
        art2.setDescription("Conseils pratiques pour réduire le stress et améliorer la performance.");
        art2.setUrl("https://example.com/articles/stress");
        art2.setType("article");
        art2.setDuration("7 min read");
        art2.setDifficulty("moyen");
        art2.setPriority(score >= 0.8 ? 1 : 2);
        articleItems.add(art2);

        articles.setItems(articleItems);
        categories.add(articles);

        // Practice category
        RecommendationCategoryDto practice = new RecommendationCategoryDto();
        practice.setCategory("Exercices");
        practice.setIcon("Exercise");
        practice.setColor("#F59E0B");

        List<RecommendationItemDto> practiceItems = new ArrayList<>();
        RecommendationItemDto ex1 = new RecommendationItemDto();
        ex1.setId("ex_1");
        ex1.setTitle("Quiz de révision rapide");
        ex1.setDescription("Un quiz de 10 questions pour renforcer les concepts clés.");
        ex1.setUrl("/courses/quick-quiz");
        ex1.setType("quiz");
        ex1.setDuration("10 min");
        ex1.setDifficulty("moyen");
        ex1.setPriority(score >= 0.5 ? 1 : 3);
        practiceItems.add(ex1);

        practice.setItems(practiceItems);
        categories.add(practice);

        resp.setCategories(categories);

        return resp;
    }
}

