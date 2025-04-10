import time
import matplotlib.pyplot as plt
import random
import string
import pandas as pd
import seaborn as sns
import os


# ------ PARTIE 1: IMPLÉMENTATION DES ALGORITHMES ------

class StringMatcher:
    """Classe abstraite pour tous les algorithmes de recherche de motifs"""
    @staticmethod
    def search(text, pattern):
        """Méthode abstraite à implémenter par les sous-classes"""
        raise NotImplementedError("Cette méthode doit être implémentée par les sous-classes")


class MorrisPratt(StringMatcher):
    """Implémentation de l'algorithme de Morris-Pratt"""
    @staticmethod
    def compute_prefix_table(pattern):
        n = len(pattern)
        prefix = [0] * n
        k = 0
        for q in range(1, n):
            while k > 0 and pattern[k] != pattern[q]:
                k = prefix[k - 1]
            if pattern[k] == pattern[q]:
                k += 1
            prefix[q] = k
        return prefix

    @staticmethod
    def search(text, pattern):
        prefix = MorrisPratt.compute_prefix_table(pattern)
        occurrences = []
        comparisons = 0
        q = 0
        
        for i in range(len(text)):
            while q > 0 and pattern[q] != text[i]:
                comparisons += 1
                q = prefix[q - 1]
            if pattern[q] == text[i]:
                comparisons += 1
                q += 1
            if q == len(pattern):
                occurrences.append(i - len(pattern) + 1)
                q = prefix[q - 1]
                
        return occurrences, comparisons


class BoyerMoore(StringMatcher):
    """Implémentation de l'algorithme de Boyer-Moore"""
    @staticmethod
    def build_bad_character_table(pattern):
        bad_char = {}
        for i in range(len(pattern)):
            bad_char[pattern[i]] = i
        return bad_char

    @staticmethod
    def search(text, pattern):
        bad_char = BoyerMoore.build_bad_character_table(pattern)
        m = len(pattern)
        n = len(text)
        occurrences = []
        comparisons = 0
        
        i = 0
        while i <= n - m:
            j = m - 1
            
            while j >= 0 and pattern[j] == text[i + j]:
                comparisons += 1
                j -= 1
                
            if j < 0:
                occurrences.append(i)
                i += 1
            else:
                comparisons += 1
                bc_shift = j - bad_char.get(text[i + j], -1)
                i += max(1, bc_shift)
                
        return occurrences, comparisons


class RabinKarp(StringMatcher):
    """Implémentation de l'algorithme de Rabin-Karp avec multi-hachage"""
    @staticmethod
    def search(text, pattern, hash_bases=[256, 101, 53], hash_mods=[101, 103, 107]):
        n = len(text)
        m = len(pattern)
        occurrences = []
        comparisons = 0
        
        # Nombre de fonctions de hachage à utiliser
        num_hashes = min(len(hash_bases), len(hash_mods))
        
        # Précalcul des puissances pour les mises à jour de hash efficaces
        h_powers = [pow(hash_bases[i], m-1) % hash_mods[i] for i in range(num_hashes)]
        
        # Calcul des hashes pour le motif
        pattern_hashes = [0] * num_hashes
        for i in range(num_hashes):
            for j in range(m):
                pattern_hashes[i] = (pattern_hashes[i] * hash_bases[i] + ord(pattern[j])) % hash_mods[i]
        
        # Calcul des hashes pour la première fenêtre du texte
        text_hashes = [0] * num_hashes
        for i in range(num_hashes):
            for j in range(m):
                text_hashes[i] = (text_hashes[i] * hash_bases[i] + ord(text[j])) % hash_mods[i]
        
        # Glissement de la fenêtre à travers le texte
        for i in range(n - m + 1):
            # Vérification des correspondances de hash
            all_hashes_match = True
            for h in range(num_hashes):
                if text_hashes[h] != pattern_hashes[h]:
                    all_hashes_match = False
                    break
            
            if all_hashes_match:
                # Vérification caractère par caractère
                match = True
                for j in range(m):
                    comparisons += 1
                    if text[i+j] != pattern[j]:
                        match = False
                        break
                if match:
                    occurrences.append(i)
            
            # Calcul des hashes pour la fenêtre suivante
            if i < n - m:
                for h in range(num_hashes):
                    text_hashes[h] = ((text_hashes[h] - ord(text[i]) * h_powers[h]) * hash_bases[h] + ord(text[i + m])) % hash_mods[h]
                    if text_hashes[h] < 0:
                        text_hashes[h] += hash_mods[h]
        
        return occurrences, comparisons


class AhoCorasick(StringMatcher):
    """Implémentation de l'algorithme d'Aho-Corasick"""
    def __init__(self, patterns):
        self.patterns = patterns
        self.num_nodes = 1
        self.edges = [{}]
        self.fail = [0]
        self.output = [set()]
        self._build()

    def _build(self):
        # Construction du trie
        for pattern in self.patterns:
            current_state = 0
            for char in pattern:
                if char not in self.edges[current_state]:
                    self.edges[current_state][char] = self.num_nodes
                    self.edges.append({})
                    self.fail.append(0)
                    self.output.append(set())
                    self.num_nodes += 1
                current_state = self.edges[current_state][char]
            self.output[current_state].add(pattern)

        # Construction de la fonction d'échec avec BFS
        queue = []
        for char, state in self.edges[0].items():
            queue.append(state)
            self.fail[state] = 0

        while queue:
            r = queue.pop(0)
            for char, state in list(self.edges[r].items()):
                queue.append(state)
                failure = self.fail[r]
                
                while failure != 0 and char not in self.edges[failure]:
                    failure = self.fail[failure]
                
                if char in self.edges[failure]:
                    self.fail[state] = self.edges[failure][char]
                else:
                    self.fail[state] = 0
                
                self.output[state].update(self.output[self.fail[state]])

    def search_multiple(self, text):
        state = 0
        all_patterns = sum([list(output) for output in self.output], [])
        occurrences = {p: [] for p in all_patterns}
        comparisons = 0
        
        for i, char in enumerate(text):
            comparisons += 1
            while state != 0 and char not in self.edges[state]:
                state = self.fail[state]
            
            if char in self.edges[state]:
                state = self.edges[state][char]
            
            for pattern in self.output[state]:
                occurrences[pattern].append(i - len(pattern) + 1)
        
        return occurrences, comparisons

    @staticmethod
    def search(text, pattern):
        ac = AhoCorasick([pattern])
        occurrences, comparisons = ac.search_multiple(text)
        return occurrences[pattern], comparisons


class CommentzWalter(StringMatcher):
    """Implémentation de l'algorithme de Commentz-Walter"""
    def __init__(self, patterns):
        self.patterns = patterns
        self.min_length = min(len(p) for p in patterns)
        self.max_length = max(len(p) for p in patterns)
        self.bad_char = self._build_shift_table()
        
    def _build_shift_table(self):
        shift = {}
        for p in self.patterns:
            m = len(p)
            for i in range(m - 1):
                shift[p[i]] = m - 1 - i
        return shift
        
    def search_multiple(self, text):
        n = len(text)
        occurrences = {p: [] for p in self.patterns}
        comparisons = 0
        
        i = self.min_length - 1
        while i < n:
            for pattern in self.patterns:
                m = len(pattern)
                if i >= m - 1:
                    j = m - 1
                    while j >= 0 and pattern[j] == text[i - (m - 1 - j)]:
                        comparisons += 1
                        j -= 1
                    
                    if j < 0:
                        occurrences[pattern].append(i - m + 1)
            
            char = text[i]
            shift = self.bad_char.get(char, self.min_length)
            i += max(1, shift)
            
        return occurrences, comparisons

    @staticmethod
    def search(text, pattern):
        cw = CommentzWalter([pattern])
        occurrences, comparisons = cw.search_multiple(text)
        return occurrences[pattern], comparisons


# ------ PARTIE 2: GÉNÉRATION DE TEXTES ET MOTIFS ------

class TextGenerator:
    """Classe pour générer différents cas de test"""
    @staticmethod
    def best_case(length_t, length_p):
        """Génère un cas favorable pour les algorithmes de recherche"""
        motif = "ab"
        P = motif * (length_p // 2)
        repetitions = (length_t // len(P))
        T = P * repetitions
        return T, P

    @staticmethod
    def worst_case(length_t, length_p):
        """Génère un cas défavorable pour les algorithmes de recherche"""
        P = "a" * (length_p - 1) + "b"
        T = "a" * length_t
        if length_t > length_p:
            T = T[:length_t - 1] + "b"
        return T, P

    @staticmethod
    def random_case(length_t, length_p):
        """Génère un cas aléatoire pour les algorithmes de recherche"""
        T = ''.join(random.choices(string.ascii_lowercase, k=length_t))
        P = ''.join(random.choices(string.ascii_lowercase, k=length_p))
        
        # Garantir qu'au moins une occurrence du motif est présente
        if length_t >= length_p:
            insert_pos = random.randint(0, length_t - length_p)
            T = T[:insert_pos] + P + T[insert_pos + length_p:]
        
        return T, P

    @staticmethod
    def custom_case():
        """Permet à l'utilisateur de définir ses propres texte et motif"""
        print("\n=== Génération d'un cas personnalisé ===")
        T = input("Entrez le texte à analyser: ")
        P = input("Entrez le motif à rechercher: ")
        return T, P


# ------ PARTIE 3: FRAMEWORK DE TEST ------

class AlgorithmTester:
    """Classe pour tester les performances des algorithmes"""
    def __init__(self):
        self.algorithms = {
            "MP": MorrisPratt,
            "BM": BoyerMoore,
            "RK": RabinKarp,
            "AC": AhoCorasick,
            "CW": CommentzWalter
        }
        
        self.generators = {
            "Best": TextGenerator.best_case,
            "Worst": TextGenerator.worst_case,
            "Random": TextGenerator.random_case,
            "Custom": TextGenerator.custom_case
        }
        
        self.test_cases = []
    
    def set_test_cases(self, test_cases):
        """Définit les cas de test à utiliser"""
        self.test_cases = test_cases
    
    def test_single_algorithm(self, algo_name, case_names=None):
        """Teste un algorithme spécifique avec les cas de test sélectionnés"""
        if case_names is None:
            case_names = list(self.generators.keys())
            
        results = []
        algorithm = self.algorithms[algo_name]
        
        for length_t, length_p in self.test_cases:
            for case_name in case_names:
                generator = self.generators[case_name]
                T, P = generator(length_t, length_p) if case_name != "Custom" else generator()
                
                print(f"\nTest de {algo_name} avec {case_name} (T={length_t if case_name != 'Custom' else len(T)}, P={length_p if case_name != 'Custom' else len(P)})")
                
                start = time.time()
                if algo_name in ["AC", "CW"]:
                    # Ces algorithmes nécessitent une instanciation
                    occurrences, comparisons = algorithm.search(T, P)
                else:
                    # Algorithmes standards
                    occurrences, comparisons = algorithm.search(T, P)
                
                end = time.time()
                exec_time = (end - start) * 1000
                
                result = {
                    "cas": case_name,
                    "taille_texte": len(T),
                    "taille_motif": len(P),
                    "algo": algo_name,
                    "comparaisons": comparisons,
                    "temps": round(exec_time, 3),
                    "occurrences": len(occurrences)
                }
                
                print(f"Résultat: {len(occurrences)} occurrences trouvées en {result['temps']} ms avec {comparisons} comparaisons")
                
                results.append(result)
        
        return results
    
    def compare_algorithms(self, algo_names, case_names=None):
        """Compare plusieurs algorithmes entre eux"""
        if case_names is None:
            case_names = list(self.generators.keys())
            
        results = []
        
        for length_t, length_p in self.test_cases:
            for case_name in case_names:
                generator = self.generators[case_name]
                T, P = generator(length_t, length_p) if case_name != "Custom" else generator()
                
                for algo_name in algo_names:
                    algorithm = self.algorithms[algo_name]
                    
                    print(f"\nTest de {algo_name} avec {case_name} (T={length_t if case_name != 'Custom' else len(T)}, P={length_p if case_name != 'Custom' else len(P)})")
                    
                    start = time.time()
                    if algo_name in ["AC", "CW"]:
                        occurrences, comparisons = algorithm.search(T, P)
                    else:
                        occurrences, comparisons = algorithm.search(T, P)
                    
                    end = time.time()
                    exec_time = (end - start) * 1000
                    
                    result = {
                        "cas": case_name,
                        "taille_texte": len(T),
                        "taille_motif": len(P),
                        "algo": algo_name,
                        "comparaisons": comparisons,
                        "temps": round(exec_time, 3),
                        "occurrences": len(occurrences)
                    }
                    
                    print(f"Résultat: {len(occurrences)} occurrences trouvées en {result['temps']} ms avec {comparisons} comparaisons")
                    
                    results.append(result)
        
        return results


# ------ PARTIE 4: VISUALISATION DES RÉSULTATS ------

class ResultVisualizer:
    """Classe pour l'affichage et la visualisation des résultats"""
    @staticmethod
    def print_results(results):
        """Affiche les résultats sous forme de tableau"""
        if not results:
            print("Aucun résultat à afficher.")
            return
            
        print("\n" + "=" * 80)
        print(f"{'Algo':<5} | {'Cas':<8} | {'Texte':<6} | {'Motif':<6} | {'Comp.':<10} | {'Temps (ms)':<10} | {'Occur.'}")
        print("-" * 80)
        for r in results:
            print(f"{r['algo']:<5} | {r['cas']:<8} | {r['taille_texte']:<6} | {r['taille_motif']:<6} | {r['comparaisons']:<10} | {r['temps']:<10} | {r['occurrences']}")
        print("=" * 80 + "\n")

    @staticmethod
    def plot_comparisons(results, title_prefix="Comparaison", show_plots=True, save_dir=None):
        """Visualise les résultats sous forme de graphiques"""
        if not results:
            print("Aucun résultat à visualiser.")
            return
            
        df = pd.DataFrame(results)
        
        for metric in ["comparaisons", "temps"]:
            plt.figure(figsize=(12, 6))
            sns.barplot(data=df, x="cas", y=metric, hue="algo")
            plt.title(f"{title_prefix} - {metric}")
            plt.ylabel(metric.capitalize())
            plt.xlabel("Cas de test")
            plt.legend(title="Algorithme")
            plt.tight_layout()
            
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                filename = f"{title_prefix.lower().replace(' ', '_')}_{metric}.png"
                filepath = os.path.join(save_dir, filename)
                plt.savefig(filepath)
                print(f"Graphique enregistré: {filepath}")
            
            if show_plots:
                plt.show()
            else:
                plt.close()
    
    @staticmethod
    def export_to_csv(results, filename="resultats.csv"):
        """Exporte les résultats vers un fichier CSV"""
        if not results:
            print("Aucun résultat à exporter.")
            return
            
        df = pd.DataFrame(results)
        df.to_csv(filename, index=False)
        print(f"Résultats exportés vers {filename}")


# ------ PARTIE 5: INTERFACE UTILISATEUR ------

class UserInterface:
    """Interface utilisateur pour le framework de test"""
    def __init__(self):
        self.tester = AlgorithmTester()
        self.visualizer = ResultVisualizer()
        self.all_results = []
        
    def clear_screen(self):
        """Nettoie l'écran de la console"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self):
        """Affiche l'en-tête de l'application avec un logo ASCII"""
        self.clear_screen()
        print("=" * 80)
        print("""
        ╔══════════════════════════════════════════════════════════════════════════╗
        ║                          ┌─┐┌┬┐┌┬┐┌─┐┌─┐┌┬┐                              ║
        ║                          ├┤ │││ │ │ │││  │                               ║
        ║                          └  ┴ ┴ ┴ └─┘└─┘ ┴                               ║
        ║                                                                          ║     
        ║                     S.MOKEDDEM       O.BOUSSAHLA                         ║                                                
        ║                                                                          ║
        ║                BIOINFORMATIQUE PATTERN MATCHER - USTHB                   ║
        ║                                                                          ║
        ║             Framework d'analyse pour séquences génomiques                ║
        ╚══════════════════════════════════════════════════════════════════════════╝
    """)
    print("=" * 80)
    def get_choice(self, prompt, options, allow_multiple=False):
        """Récupère un choix de l'utilisateur parmi des options"""
        while True:
            print(f"\n{prompt}")
            for i, option in enumerate(options, 1):
                print(f"{i}. {option}")
            
            try:
                if allow_multiple:
                    print("\nEntrez les numéros séparés par des espaces ou 'all' pour tout sélectionner")
                    user_input = input("> ").strip()
                    
                    if user_input.lower() == 'all':
                        return list(range(len(options)))
                    
                    choices = [int(c) - 1 for c in user_input.split()]
                    if all(0 <= c < len(options) for c in choices):
                        return choices
                else:
                    choice = int(input("> ").strip()) - 1
                    if 0 <= choice < len(options):
                        return choice
                
                print("Choix non valide.")
            except ValueError:
                print("Veuillez entrer un nombre valide.")
    
    def configure_test_cases(self):
        """Configure les cas de test à utiliser"""
        self.print_header()
        print("\n=== CONFIGURATION DES CAS DE TEST ===")
        
        test_cases = []
        
        print("\nVoulez-vous utiliser les cas de test par défaut ou personnalisés?")
        choice = self.get_choice("Sélectionnez une option:", ["Cas par défaut", "Cas personnalisés"])
        
        if choice == 0:  # Cas par défaut
            test_cases = [(100, 10), (1000, 100)]
        else:  # Cas personnalisés
            num_cases = int(input("\nCombien de cas de test souhaitez-vous définir? "))
            for i in range(num_cases):
                print(f"\nCas de test #{i+1}:")
                text_length = int(input("Longueur du texte: "))
                pattern_length = int(input("Longueur du motif: "))
                test_cases.append((text_length, pattern_length))
        
        self.tester.set_test_cases(test_cases)
        print("\nCas de test configurés avec succès!")
        input("\nAppuyez sur Entrée pour continuer...")
    
    def run_single_algorithm_test(self):
        """Exécute le test d'un seul algorithme"""
        self.print_header()
        print("\n=== TEST D'UN SEUL ALGORITHME ===")
        
        algo_names = list(self.tester.algorithms.keys())
        algo_full_names = ["Morris-Pratt", "Boyer-Moore", "Rabin-Karp (3 hashes)", "Aho-Corasick", "Commentz-Walter"]
        algo_choices = [f"{a} - {b}" for a, b in zip(algo_names, algo_full_names)]
        
        algo_idx = self.get_choice("Choisissez un algorithme à tester:", algo_choices)
        algo_name = algo_names[algo_idx]
        
        case_names = list(self.tester.generators.keys())
        case_idx_list = self.get_choice("Choisissez les cas de test à utiliser:", case_names, allow_multiple=True)
        selected_cases = [case_names[idx] for idx in case_idx_list]
        
        print(f"\nLancement des tests pour {algo_full_names[algo_idx]} avec les cas: {', '.join(selected_cases)}")
        results = self.tester.test_single_algorithm(algo_name, selected_cases)
        
        self.visualizer.print_results(results)
        
        plot_choice = self.get_choice("Voulez-vous visualiser les résultats?", ["Oui", "Non"])
        if plot_choice == 0:
            self.visualizer.plot_comparisons(results, f"Résultats de {algo_full_names[algo_idx]}")
        
        self.all_results.extend(results)
        
        input("\nAppuyez sur Entrée pour continuer...")
        return results
    
    def run_comparison_test(self):
        """Exécute un test de comparaison entre plusieurs algorithmes"""
        self.print_header()
        print("\n=== COMPARAISON ENTRE ALGORITHMES ===")
        
        algo_names = list(self.tester.algorithms.keys())
        algo_full_names = ["Morris-Pratt", "Boyer-Moore", "Rabin-Karp (3 hashes)", "Aho-Corasick", "Commentz-Walter"]
        algo_choices = [f"{a} - {b}" for a, b in zip(algo_names, algo_full_names)]
        
        algo_idx_list = self.get_choice("Choisissez les algorithmes à comparer:", algo_choices, allow_multiple=True)
        selected_algos = [algo_names[idx] for idx in algo_idx_list]
        
        case_names = list(self.tester.generators.keys())
        case_idx_list = self.get_choice("Choisissez les cas de test à utiliser:", case_names, allow_multiple=True)
        selected_cases = [case_names[idx] for idx in case_idx_list]
        
        print(f"\nLancement de la comparaison entre {', '.join([algo_full_names[idx] for idx in algo_idx_list])}")
        print(f"Cas de test: {', '.join(selected_cases)}")
        
        results = self.tester.compare_algorithms(selected_algos, selected_cases)
        
        self.visualizer.print_results(results)
        
        plot_choice = self.get_choice("Voulez-vous visualiser les résultats?", ["Oui", "Non"])
        if plot_choice == 0:
            algo_names_display = " vs ".join([algo_names[idx] for idx in algo_idx_list])
            self.visualizer.plot_comparisons(results, f"Comparaison {algo_names_display}")
        
        self.all_results.extend(results)
        
        input("\nAppuyez sur Entrée pour continuer...")
        return results
    
    def view_all_results(self):
        """Affiche tous les résultats accumulés"""
        self.print_header()
        print("\n=== TOUS LES RÉSULTATS ===")
        
        if not self.all_results:
            print("\nAucun résultat disponible. Exécutez des tests d'abord.")
            input("\nAppuyez sur Entrée pour continuer...")
            return
        
        self.visualizer.print_results(self.all_results)
        
        export_choice = self.get_choice("Voulez-vous exporter les résultats?", ["Oui", "Non"])
        if export_choice == 0:
            filename = input("\nNom du fichier CSV (resultats.csv par défaut): ") or "resultats.csv"
            self.visualizer.export_to_csv(self.all_results, filename)
        
        plot_choice = self.get_choice("Voulez-vous visualiser les résultats?", ["Oui", "Non"])
        if plot_choice == 0:
            self.visualizer.plot_comparisons(self.all_results, "Tous les résultats")
        
        input("\nAppuyez sur Entrée pour continuer...")
    
    def run_menu(self):
        """Exécute le menu principal de l'application"""
        while True:
            self.print_header()
            print("\n=== MENU PRINCIPAL ===")
            
            choice = self.get_choice("Que souhaitez-vous faire?", [
                "Configurer les cas de test",
                "Tester un algorithme",
                "Comparer plusieurs algorithmes",
                "Voir tous les résultats",
                "Quitter"
            ])
            
            if choice == 0:
                self.configure_test_cases()
            elif choice == 1:
                self.run_single_algorithm_test()
            elif choice == 2:
                self.run_comparison_test()
            elif choice == 3:
                self.view_all_results()
            elif choice == 4:
                print("\nAu revoir!")
                break


# ------ PARTIE 6: FONCTION PRINCIPALE ------

def main():
    """Fonction principale exécutant l'interface utilisateur"""
    ui = UserInterface()
    ui.run_menu()


# Lancement du programme si exécuté directement
if __name__ == "__main__":
    main()