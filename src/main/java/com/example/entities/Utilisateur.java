package com.example.entities;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.Date;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Entity
@Inheritance(strategy = InheritanceType.JOINED)
public class Utilisateur {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    protected int id;
    protected String nom;
    protected String prenom;
    protected String email;
    protected String password;
    protected String adresse;
    protected String telephone;
    protected String genre;

    @Enumerated(EnumType.STRING)
    protected Role role;
    @Temporal(TemporalType.DATE)
    protected Date dateNaissance;


}
