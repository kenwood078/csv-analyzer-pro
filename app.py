import streamlit as st
import pandas as pd
import chardet
from io import BytesIO
import csv
import plotly.express as px
import plotly.graph_objects as go

# ---------- Page Configuration ----------
st.set_page_config(
	page_title='CSV Analyzer Pro',
	page_icon='📊',
	layout='wide',
	initial_sidebar_state='collapsed'
)


# ---------- Helper Functions with Caching ----------

@st.cache_data(show_spinner='Определение кодировки...')
def detect_encoding(file_bytes: bytes) -> str:
	"""Определяет кодировку CSV-файла на основе первых 10 КБ.

	Использует библиотеку chardet. Если обнаружена 'ascii',
	возвращается 'utf-8' (ASCII является подмножеством UTF-8).
	При невозможности определить кодировку выбрасывается ValueError.

	Args:
		file_bytes: Байтовое содержимое загруженного файла.

	Returns:
		Строка с названием кодировки (например, 'utf-8', 'windows-1251').

	Raises:
		ValueError: Если кодировку не удалось определить (encoding is None).
	"""
	sample_bytes = file_bytes[:10000]
	result = chardet.detect(sample_bytes)
	encoding = result.get('encoding')

	# ASCII – строгое подмножество UTF-8, безопасно для дальнейшей обработки
	if encoding == 'ascii':
		return 'utf-8'
	if encoding is None:
		raise ValueError('Не удалось определить кодировку файла!')
	return encoding


@st.cache_data(show_spinner='Определение разделителя...')
def detect_separator(file_bytes: bytes, encoding: str) -> str:
	"""Определяет разделитель CSV с помощью csv.Sniffer.

	При неудаче применяется fallback на эвристику:
	выбирается символ-разделитель с максимальным числом вхождений
	среди распространённых вариантов.

	Args:
		file_bytes: Байтовое содержимое файла.
		encoding: Кодировка, полученная на предыдущем шаге.

	Returns:
		Строка с определённым разделителем (напр., ',', ';', '\t', '|').
	"""
	sample_bytes = file_bytes[:10000]
	sample_text = sample_bytes.decode(encoding, errors='replace')

	try:
		# Sniffer анализирует первые несколько строк
		lines = sample_text.splitlines()[:5]
		sniffer = csv.Sniffer()
		dialect = sniffer.sniff('\n'.join(lines))
		separator = dialect.delimiter
	except csv.Error:
		# fallback: выбор разделителя с максимальным числом вхождений
		separators = [',', ';', '\t', '|']
		separator = max(separators, key=lambda s: sample_text.count(s))

	return separator


@st.cache_data(show_spinner='Загрузка и обработка CSV...')
def load_csv(file_bytes: bytes, encoding: str, separator: str) -> pd.DataFrame:
	"""Загружает DataFrame из байтового представления CSV.

	Выполняет парсинг дат, а также поиск столбцов,
	которые можно преобразовать (эвристика по 80%).

	Args:
		file_bytes: Байтовый массив CSV.
		encoding: Кодировка файла.
		separator: Разделитель.

	Returns:
		pd.DataFrame с загруженными и обработанными данными.

	Raises:
		pd.errors.EmptyDataError: Если файл не содержит данных.
	"""
	try:
		df = pd.read_csv(
			BytesIO(file_bytes),
			encoding=encoding,
			sep=separator,
			parse_dates=True,
			# na_values=[] оставлены по умолчанию (покрывают основные случаи)
		)
	except pd.errors.EmptyDataError:
		raise pd.errors.EmptyDataError('Файл пустой!')

	# Постобработка: поиск скрытых дат в текстовых колонках
	for col in df.select_dtypes(include=['object']).columns:
		try:
			sample = df[col].dropna().head(100)
			if not sample.empty:
				converted = pd.to_datetime(sample, errors='coerce')
				# Если более 80% успешно преобразовались, меняем тип столбца
				if converted.notna().mean() > 0.8:
					df[col] = pd.to_datetime(df[col], errors='coerce')
		except Exception:
			# оставляем как есть при любой ошибке
			pass

	return df


@st.cache_data(show_spinner='Подготовка файла для скачивания...')
def convert_df_to_csv(df: pd.DataFrame) -> bytes:
	"""Конвертирует DataFrame в байтовое представление CSV.

	Args:
		df: Исходный DataFrame.

	Returns:
		Байты в формате CSV (UTF-8).
	"""
	return df.to_csv(index=False).encode('utf-8')


@st.cache_data(show_spinner='Подготовка графика для скачивания...')
def get_figure_bytes(fig: go.Figure) -> bytes:
	"""Конвертирует Plotly Figure в PNG байты.

	Args:
		fig: Plotly Figure.

	Returns:
		Байты PNG изображения.
	"""
	return fig.to_image(format='png', scale=3)


def render_plot_with_download(fig: go.Figure, filename: str,
                              label: str = '⬇️ Скачать график (PNG)') -> None:
	"""
	Отрисовывает Plotly график и добавляет кнопку скачивания PNG.
	При отсутствии kaleido выводит предупреждение и не показывает кнопку.

	Args:
		fig: Plotly Figure для отображения.
		filename: Имя файла для скачивания.
		label: Текст на кнопке скачивания.
	"""
	st.plotly_chart(fig, use_container_width=True)
	try:
		img_bytes = get_figure_bytes(fig)
		st.download_button(
			label=label,
			data=img_bytes,
			file_name=filename,
			mime='image/png'
		)
	except Exception as e:
		st.warning(f'Не удалось сгенерировать PNG для скачивания.'
		           f'Убедитесь, что установлен kaleido. ({e})')


@st.cache_data(show_spinner='Вычисление статистики...')
def compute_numeric_stats(df: pd.DataFrame, column: str) -> dict:
	"""Вычисляет базовые статистики для числового столбца.

	Args:
		df: DataFrame с данными.
		column: Имя числового столбца.

	Returns:
		Словарь со средним, медианой, стандартным отклонением,
		минимумом, максимумом и количеством непустых значений.
	"""
	series = df[column].dropna()
	return {
		'Mean': series.mean(),
		'Median': series.median(),
		'Std Dev': series.std(),
		'Min': series.min(),
		'Max': series.max(),
		'Count': len(series)
	}


@st.cache_resource(show_spinner='Построение графика...')
def generate_distribution_plot(df: pd.DataFrame, column: str, bins: int) -> go.Figure:
	"""Строит гистограмму распределения с наложенным box plot.

	Args:
		df: DataFrame с данными.
		column: Имя числового столбца.
		bins: Количество бинов гистограммы.

	Returns:
		Plotly Figure (гистограмма + box plot).
	"""
	fig = px.histogram(
		df, x=column,
		marginal='box',
		title=f'Распределение: {column}',
		nbins=bins,
		opacity=0.8
	)
	fig.update_layout(template='plotly_white')
	return fig


@st.cache_resource(show_spinner='Построение графика...')
def generate_line_chart(df: pd.DataFrame, x_col: str, y_col: str) -> go.Figure:
	"""Строит линейный график для двух столбцов.

	Args:
		df: DataFrame.
		x_col: Имя столбца для оси X (может быть любым типом).
		y_col: Имя числового столбца для оси Y.

	Returns:
		Plotly Figure с линейным графиком.
	"""
	fig = px.line(
		df, x=x_col, y=y_col,
		title=f'Линейный график: {y_col} от {x_col}',
		render_mode='webgl'
	)
	fig.update_layout(xaxis_title=x_col, yaxis_title=y_col, template='plotly_white')
	return fig


@st.cache_resource(show_spinner='Построение графика...')
def generate_scatter_chart(df: pd.DataFrame, x_col: str, y_col: str,
                           color_col: str | None = None) -> go.Figure:
	"""Строит диаграмму рассеяния с опциональной цветовой группировкой.

	Args:
		df: DataFrame.
		x_col: Имя числового столбца для оси X.
		y_col: Имя числового столбца для оси Y.
		color_col: Имя столбца для цветового кодирования (или None).

	Returns:
		Plotly Figure – диаграмма рассеяния.
	"""
	fig = px.scatter(
		df, x=x_col, y=y_col,
		color=color_col,
		title=f'Диаграмма рассеяния: {y_col} vs {x_col}', opacity=0.8,
		render_mode='webgl'
	)
	fig.update_layout(xaxis_title=x_col, yaxis_title=y_col, template='plotly_white')
	return fig


@st.cache_resource(show_spinner='Построение графика...')
def generate_bar_chart(df: pd.DataFrame, x_col: str, y_col: str,
                       color_col: str | None = None,
                       agg_func: str = 'sum',
                       barmode: str = 'group') -> go.Figure:
	"""Строит столбчатую диаграмму с заданной агрегацией и режимом отображения.

	Args:
		df: DataFrame.
		x_col: Категориальный столбец для оси X.
		y_col: Числовой столбец для агрегации.
		color_col: Опциональный столбец для дополнительной группировки.
		agg_func: Агрегирующая функция ('sum', 'mean', 'median', 'count').
		barmode: Режим отображения столбцов('group', 'stack', 'relative', 'overlay').

	Returns:
		Plotly Figure – столбчатая диаграмма.
	"""
	group_cols = [x_col] if color_col is None else [x_col, color_col]
	grouped = df.groupby(group_cols)[y_col].agg(agg_func).reset_index()

	fig = px.bar(
		grouped, x=x_col, y=y_col,
		color=color_col,
		barmode=barmode,
		title=f'Столбчатая диаграмма {agg_func} {y_col} по {x_col}'
	)
	fig.update_layout(xaxis_title=x_col, yaxis_title=y_col, template='plotly_white')
	return fig


@st.cache_resource(show_spinner='Построение матрицы...')
def generate_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
	"""Строит тепловую карту корреляции для числовых столбцов.

	Args:
	    df: DataFrame только с числовыми колонками.

	Returns:
	    Plotly Figure с тепловой картой корреляций.
	"""
	corr = df.corr()
	fig = px.imshow(
		corr,
		text_auto='.2f',
		aspect='auto',
		title='Матрица корреляции числовых признаков',
		color_continuous_scale='RdBu_r'
	)
	fig.update_layout(template='plotly_white')
	return fig


@st.cache_resource(show_spinner='Построение матрицы...')
def generate_scatter_matrix(df: pd.DataFrame) -> go.Figure:
	"""Строит матрицу диаграмм рассеяния для выбранных столбцов.

	Args:
	    df: DataFrame с числовыми колонками.

	Returns:
	    Plotly Figure с матрицей scatter-графиков.
	"""
	fig = px.scatter_matrix(
		df,
		title='Матрица рассеяния',
		opacity=0.8,
	)
	fig.update_layout(template='plotly_white', height=1000)
	return fig


# ---------- Main Application ----------

def main():
	"""Точка входа в приложение. Организует загрузку CSV и интерфейс анализа."""
	st.title('📊CSV-Анализатор')
	st.markdown("""
    Загрузите любой CSV-файл и изучите его содержимое с помощью интерактивной статистики и визуализации.
    Приложение автоматически определяет кодировку и разделители, обрабатывает даты и кэширует операции для повышения скорости.
    """)

	with st.sidebar:
		st.header('📂 Источник данных')
		upload_file = st.file_uploader(
			'Загрузите CSV-файл',
			type=['csv', 'txt'],
			help='Поддерживаются любые разделители и кодировки',
			key='file_uploader'
		)
		if upload_file is not None:
			file_bytes = upload_file.getvalue()
			st.session_state.file_bytes = file_bytes
			# Автоматическое определение параметров и загрузки
			try:
				encoding = detect_encoding(file_bytes)
				sep = detect_separator(file_bytes, encoding)
				df = load_csv(file_bytes, encoding, sep)

				# Сохраняем в сессию при успехе
				st.session_state.df = df
				st.session_state.encoding = encoding
				st.session_state.sep = sep
			except ValueError as e:
				st.error(f'Ошибка: {str(e)}')
				st.stop()
			except pd.errors.EmptyDataError:
				st.error('Файл пуст')
				st.stop()
			except Exception as e:
				st.error(f'Ошибка загрузки: {str(e)}')
				st.stop()

			st.success(f'✅ Загружено {df.shape[0]:,} строк × {df.shape[1]} столбцов')
		else:
			st.info('👆 Пожалуйста, загрузите CSV-файл для начала работы')
			st.stop()

		# Расширенные параметры (ручное переопределение)
		if st.session_state.df is not None:
			st.write(f'🔍 Обнаружена кодировка: `{st.session_state.encoding}`')
			st.write(f'🔍 Обнаружен разделитель: `{st.session_state.sep}`')

			with st.expander('⚙️ Расширенные параметры'):
				manual_enc = st.text_input('Кодировка', value=st.session_state.encoding)
				manual_sep = st.text_input('Разделитель', value=st.session_state.sep)
				if st.button('Применить'):
					try:
						df = load_csv(
							st.session_state.file_bytes,
							encoding=manual_enc or st.session_state.encoding,
							separator=manual_sep or st.session_state.sep
						)
						st.session_state.df = df
						st.session_state.encoding = manual_enc or st.session_state.encoding
						st.session_state.sep = manual_sep or st.session_state.sep
					except Exception as e:
						st.error(f'Ошибка с расширенными параметрами: {e}')

	# Основная рабочая область
	df = st.session_state.df
	tab1, tab2, tab3, tab4 = st.tabs([
		'📋 Предварительный просмотр', '📊 Статистика', '📈 Графики', '🔬 Расширенный анализ'
	])

	# ─── Вкладка 1: Просмотр и редактирование данных ───
	with tab1:
		st.subheader('Загруженные данные')
		with st.expander('Типы столбцов и информация'):
			col_info = pd.DataFrame({
				'Column': df.columns,
				'Type': df.dtypes.astype(str),
				'Non-Null Count': df.count().values,
				'Null %': (df.isna().sum() / len(df) * 100).round(2).astype(str) + '%'
			})
			st.dataframe(col_info, use_container_width=True, hide_index=True)

		select_columns = st.multiselect(
			'Выберите столбцы для отображения',
			options=df.columns,
			default=df.columns
		)

		edited_df = st.data_editor(
			df[select_columns],
			use_container_width=True,
			num_rows='dynamic'
		)

		csv_data = convert_df_to_csv(edited_df)
		st.download_button(
			label='📥 Скачать CSV-файл (после обработки)',
			data=csv_data,
			file_name='processed_data.csv',
			mime='text/csv'
		)

	# ─── Вкладка 2: Статистика ───
	with tab2:
		st.subheader('Статистический анализ')
		with st.expander('📋 Полная статистика'):
			desc_df = df.describe(include='all').transpose()
			# Приводим бесконечности к NaN, чтобы избежать OverflowError в PyArrow,
			# который не поддерживает inf при конвертации в Arrow-таблицу.
			desc_df = desc_df.replace([float('inf'), float('-inf')], float('nan'))
			st.dataframe(desc_df, use_container_width=True)

		numeric_cols = df.select_dtypes(include=['number']).columns
		if numeric_cols.empty:
			st.warning('В наборе данных не обнаружено числовых столбцов')
		else:
			selected_stat_col = st.selectbox(
				'Выберите числовой столбец для анализа:',
				numeric_cols, key='stat_col'
			)
			bins = st.number_input('Введите количество bin', value=30,
			                       min_value=1, max_value=1000, key='tab2_bins')
			if selected_stat_col:
				stats = compute_numeric_stats(df, selected_stat_col)
				if stats:
					col1, col2, col3 = st.columns(3)
					col1.metric('Mean', f"{stats['Mean']:.4f}")
					col2.metric('Median', f"{stats['Median']:.4f}")
					col3.metric('Std Deviation', f"{stats['Std Dev']:.4f}")
					col4, col5, col6 = st.columns(3)
					col4.metric('Minimum', f"{stats['Min']:.4f}")
					col5.metric('Maximum', f"{stats['Max']:.4f}")
					col6.metric('Valid Count', f"{stats['Count']:,}")

					fig = generate_distribution_plot(df, selected_stat_col, bins)
					render_plot_with_download(fig, f'dist_{selected_stat_col}.png',
					                          f'⬇️ Скачать график распределения (PNG)')

	# ─── Вкладка 3: Интерактивные графики ───
	with tab3:
		st.subheader('Интерактивное построение графиков')
		container = st.container(border=True)

		plot_type = container.radio(
			'Выберите тип графика:',
			['Line Chart', 'Scatter Plot', 'Bar Chart'],
			horizontal=True,
			help='Line Chart - линейный график, '
			     'Scatter Plot - диаграмма рассеяния, '
			     'Bar Chart - столбчатая диаграмма с агрегацией.'
		)

		# Все столбцы
		all_columns = df.columns.tolist()
		# Числовые столбцы
		numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
		# Категориальные колонки: object, category, bool + даты
		categorical_cols = df.select_dtypes(
			include=['object', 'category', 'bool', 'datetime']).columns.tolist()

		# --- Line Chart ---
		if plot_type == 'Line Chart':
			if not numeric_cols:
				container.warning('В наборе данных не обнаружено числовых столбцов')
			else:
				x_col = container.selectbox('X-axis', all_columns, key='line_x')
				y_col = container.selectbox('Y-axis', numeric_cols, key='line_y')
				if container.button(f'Создать {plot_type}', type='primary'):
					fig = generate_line_chart(df, x_col, y_col)
					render_plot_with_download(fig, f'line_{y_col}_vs_{x_col}.png',
					                          f'⬇️ Скачать {plot_type} (PNG)')

		# --- Scatter Plot ---
		if plot_type == 'Scatter Plot':
			if len(numeric_cols) < 2:
				container.warning('Для Scatter Plot нужно минимум два числовых столбца (X и Y)')
			else:
				x_col = container.selectbox('X-axis', numeric_cols, key='scatter_x')
				y_col = container.selectbox('Y-axis', numeric_cols, key='scatter_y')
				color_col = container.selectbox(
					'Color by (опционально)',
					['None'] + all_columns,
					key='scatter_color',
					help='Дополнительная группировка по цвету. '
					     'Выберите "None", чтобы не использовать.'
				)
				if container.button(f'Создать {plot_type}', type='primary'):
					fig = generate_scatter_chart(
						df, x_col, y_col,
						color_col=None if color_col == 'None' else color_col
					)
					render_plot_with_download(fig, f'scatter_{y_col}_vs_{x_col}.png',
					                          f'⬇️ Скачать {plot_type} (PNG)')

		# --- Bar Chart ---
		if plot_type == 'Bar Chart':
			if not categorical_cols:
				container.warning('Нет категориальных столбцов для оси X')
			elif not numeric_cols:
				container.warning('Нет числовых столбцов для оси Y')
			else:
				x_col = container.selectbox('X-axis (категория)',
				                            categorical_cols,
				                            key='bar_x',
				                            help='Категориальный столбец для группировки'
				                            )
				y_col = container.selectbox('Y-axis (число)',
				                            numeric_cols,
				                            key='bar_y',
				                            help='Числовой столбец, значения которого будут агрегированы',
				                            )
				agg_func = container.radio('Агрегация:',
				                           ['sum', 'mean', 'median', 'count'],
				                           horizontal=True, key='bar_agg',
				                           help='Выбор агрегирующей функции для Y внутри категории X'
				                           )
				barmode = container.radio('Режим отображения столбцов',
				                          ['group', 'stack', 'relative', 'overlay'],
				                          horizontal=True,
				                          index=0,
				                          key='bar_barmode',
				                          help='group - рядом, stack - накопление, relative - доли, overlay - наложение'
				                          )
				# Исключаем конфликт: color не должен совпадать с x_col
				color_options = ['None'] + [col for col in all_columns if col != x_col]
				color_col = container.selectbox('Color by (опционально)',
				                                color_options,
				                                key='bar_color',
				                                help='Дополнительная группировка по цвету. '
				                                     'Выберите "None", чтобы не использовать.'
				                                )
				if container.button(f'Создать {plot_type}', type='primary'):
					fig = generate_bar_chart(
						df, x_col, y_col,
						color_col=None if color_col == 'None' else color_col,
						agg_func=agg_func,
						barmode=barmode
					)
					render_plot_with_download(fig, f'bar_{agg_func}_{y_col}_by_{x_col}.png',
					                          f'⬇️ Скачать {plot_type} (PNG)')

	# ─── Вкладка 4: Расширенный корреляционный анализ ───
	with tab4:
		st.subheader('Расширенный корреляционный анализ')
		numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
		if len(numeric_cols) < 2:
			st.warning('Для корреляционного анализа нужно минимум два числовых столбца')
		else:
			# Тепловая карта корреляции
			fig = generate_correlation_heatmap(df[numeric_cols])
			render_plot_with_download(fig, f'corr_matrix.png',
			                          f'⬇️ Скачать матрицу корреляции (PNG)')

			# Матрица рассеяния
			show_matrix = st.checkbox(
				'Показать матрицу рассеяния (может быть медленно для больших данных)',
				value=False
			)
			if show_matrix:
				# Выбор столбцов – по умолчанию первые 4 (или меньше)
				default_cols = numeric_cols[:min(4, len(numeric_cols))]
				selected_cols = st.multiselect(
					'Выберите столбцы для парного графика',
					options=numeric_cols,
					default=default_cols
				)
				if selected_cols:
					fig = generate_scatter_matrix(df[selected_cols])
					render_plot_with_download(fig, f'scatter_matrix.png',
					                          f'⬇️ Скачать матрицу рассеяния (PNG)')
				else:
					st.warning('Выберите хотя бы один столбец')


if __name__ == "__main__":
	main()
