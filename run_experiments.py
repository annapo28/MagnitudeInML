import sys
import argparse
from experiments.synthetic import (
    test_block_diagonal_structure,
    compare_magnitude_methods_with_curvature as compare_magnitude_methods, 
    compare_clustering_methods
)
from experiments.scalability import scalability_test

def main():
    parser = argparse.ArgumentParser(description='Запуск экспериментов с гибридным подходом к магнитуде')
    parser.add_argument('--test', type=str, choices=['block', 'magnitude', 'clustering', 'scalability', 'all'],
                       default='all', help='Какой тест запустить')
    args = parser.parse_args()
    
    print("=" * 70)
    print("Гибридный подход к вычислению магнитуды метрических пространств")
    print("Сравнение с методами Андреевой и классическими алгоритмами")
    print("=" * 70)
    
    if args.test in ['block', 'all']:
        test_block_diagonal_structure()
    
    if args.test in ['magnitude', 'all']:
        compare_magnitude_methods() 
    
    if args.test in ['clustering', 'all']:
        compare_clustering_methods()
    
    if args.test in ['scalability', 'all']:
        scalability_test()
    
    print("Все эксперименты завершены!")

if __name__ == "__main__":
    main()