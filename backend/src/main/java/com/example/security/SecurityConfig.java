package com.example.security;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.config.Customizer;
import org.springframework.security.config.annotation.authentication.configuration.AuthenticationConfiguration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.web.SecurityFilterChain;

@Configuration
public class SecurityConfig {

    @Bean
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder();
    }

    @Bean
    public SecurityFilterChain securityFilterChain(HttpSecurity http) throws Exception {
        http
                .csrf(csrf -> csrf.disable())
                .cors(Customizer.withDefaults())
                .headers(headers -> headers.frameOptions(frame -> frame.disable())) // Allow H2 Console frames
                .authorizeHttpRequests(auth -> auth
                        .requestMatchers("/h2-console/**").permitAll() // Allow H2 Console access
                        .requestMatchers("/auth/**").permitAll()
                        .requestMatchers("/dashboard/**").permitAll() // autorise l'accueil mobile (pas de token)
                        .requestMatchers("/courses/**").permitAll() // liste des cours
                        .requestMatchers("/classes/**").permitAll() // classes & modules (tests sans auth)
                        .requestMatchers("/at-risk/**").permitAll() // etudiants à risque (mock)
                        .requestMatchers("/assignments/**").permitAll() // liste des devoirs
                        .requestMatchers("/api/dashboard/**").permitAll()
                        .requestMatchers("/recommendations/**").permitAll()
                        .requestMatchers("/api/auth/**").permitAll()
                        .requestMatchers("/grades/**").permitAll() // notes
                        .anyRequest().authenticated()
                );
        return http.build();
    }

    @Bean
    public AuthenticationManager authenticationManager(AuthenticationConfiguration config) throws Exception {
        return config.getAuthenticationManager();
    }
}
