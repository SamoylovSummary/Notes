"""
Вспомогательные общие функции.
"""

#---------------------------------------------------------------------------------------------------

"""
Проверяем, что модуль переимпортировался в IPython notebook.
"""
print('toy_net.common reload')

#---------------------------------------------------------------------------------------------------


def indent(text, prefix='  '):
    """Добавляет в начало каждой строки текста text сдвиг prefix"""
    lines = text.split('\n')
    lines = [prefix + line for line in lines]
    # Если последняя строка была пустая, не сдвигаем её.
    if lines[-1] == prefix:
        lines[-1] = ''
    return '\n'.join(lines)

#---------------------------------------------------------------------------------------------------