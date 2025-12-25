package com.example.dto;

import com.fasterxml.jackson.annotation.JsonAlias;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class AtRiskStudentDto {
    @JsonAlias({ "id", "Id", "student_id", "studentId", "idStudent" })
    private Integer id;

    @JsonAlias({ "nom", "name", "fullname", "full_name", "username" })
    private String nom;

    @JsonAlias({ "classe", "class", "className", "classe_name" })
    private String classe;

    @JsonAlias({ "modules", "module", "modules_list", "modulesList" })
    private String modules;

    @JsonAlias({ "risque", "risk", "score", "risk_score", "riskScore" })
    private Integer risque;

    @JsonAlias({ "derniereConnexion", "lastConnection", "last_seen", "lastSeen" })
    private String derniereConnexion;

    @JsonAlias({ "niveau", "level", "status" })
    private String niveau;

    @JsonAlias({ "avatar", "photo", "image", "avatar_url" })
    private String avatar;
}